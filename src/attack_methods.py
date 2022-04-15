import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models import *
#from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math
from PIL import Image
import pickle

import pudb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Attack_None(nn.Module):
    
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, vit, featureExtractor, config):
        super(Attack_PGD, self).__init__()

        self.vit = vit
        self.featureExtractor = featureExtractor
        #for p in self.parameters():
        #    p.requires_grad = False
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_labels = config['num_labels']
        self.top_k = config['top_k']
        self.patch_num = config['patch_num']
        self.image_size = config['image_size']
        self.attack_flag = config['attack_flag']
        self.num_step = config['num_step']
        self.epsilon = 0.3
        self.patch_size = int(self.image_size / self.patch_num) #16
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        #self.attack_w = nn.Parameter(torch.FloatTensor(torch.rand(3, self.image_size, self.image_size)))   #[3, 224, 224]
        self.attack_w = nn.Parameter(torch.FloatTensor(torch.rand(self.patch_num * self.patch_num, config['embedding_d'])*2-1)).to(device)   #[14 * 14, 768]

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):
        aux_vit = pickle.loads(pickle.dumps(self.vit))

        for name, value in aux_vit.named_parameters():
            if "patch_embeddings" in name:
                value.requires_grad = True
        
        x = inputs
        #x = inputs.detach()
        x.requires_grad_()
        if x.grad is not None:
            x.grad.zero_()
        aux_vit.to(device)
        aux_vit.eval()
        outputs = aux_vit(x)      # logits [batch classes_size]
        #print(outputs.logits)
        #outputs.patch_embeddings.retain_grad()
        loss = self.loss_func(outputs.logits.view(-1, self.num_labels), targets.view(-1))
        loss.backward(retain_graph=True)
        patch_embeddings = outputs.patch_embeddings  #[n, 196, 768]
        patch_grad = patch_embeddings.grad.data.detach() #[n, 196, 768]
        #sum_patch = torch.sum(torch.sum(patch_grad, dim = 0), dim = 1) #[196]
        #vlaues, top_k_patch = sum_patch.topk(self.top_k)
        #print("patch:", patch_grad.max())
        x_grad = x.grad.data.detach()
        if "PGD" in self.attack_flag:
            for i in range(self.num_step):
                x = inputs.detach() #[batch, 3, 224, 224]
                if x.grad is not None:
                    x.grad.zero_()
                if x.grad is not None:
                    x.grad.data.fill_(0)
                aux_vit.eval()
                outputs = aux_vit(x, attack_flag=self.attack_flag, patch_grad=patch_grad, attack_w=self.attack_w, top_k=self.top_k, epsilon=self.epsilon)
                logits = outputs.logits
                loss = self.loss_func(logits.view(-1, self.num_labels), targets.view(-1))
                loss.backward(retain_graph=True)
                patch_embeddings = outputs.patch_embeddings  #[n, 196, 768]
                patch_grad_new = patch_embeddings.grad.data.detach() #[n, 196, 768]
                patch_grad = patch_grad + patch_grad_new

        else:
            #print("inputs", x.grad.data.detach().max())
            x = inputs.detach() #[batch, 3, 224, 224]
            if x.grad is not None:
                x.grad.zero_()
            if x.grad is not None:
                x.grad.data.fill_(0)
            for name, value in self.named_parameters():
                if "attack_w" in name:
                    value.requires_grad = True
                else:
                    value.requires_grad = False
            aux_vit.eval()
            #pu.db
            #x = x + self.epsilon * x_grad.sign()
            #x = torch.clamp(x, -1, 1)
            #print(self.attack_flag)
            outputs = aux_vit(x, attack_flag=self.attack_flag, patch_grad=patch_grad, attack_w=self.attack_w, top_k=self.top_k, epsilon=self.epsilon)
            logits = outputs.logits
            #targets_prob = F.softmax(logits.float(), dim=1)
            #print(targets_prob)
            #l2loss = torch.mean(torch.mul(self.attack_w, self.attack_w))
            l2loss = torch.norm(self.attack_w, p = 2)
            #print(self.attack_w)
            u = 0
            loss = u * l2loss - (1 - u) * self.loss_func(logits.view(-1, self.num_labels), targets.view(-1))
            #print(u * l2loss, loss)
            #loss.backward(retain_graph=True)
            #print(self.attack_w)

        #print(logits.max(1), targets)
        return logits, loss, outputs.patch_distribution