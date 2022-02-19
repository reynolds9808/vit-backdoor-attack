#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, OrderedDict

import datasets
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    ViTForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTPreTrainedModel,
    ViTModel,
    BeitPreTrainedModel,
    BeitModel,
    DeiTPreTrainedModel,
    DeiTModel,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

""" Fine-tuning a 🤗 Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.16.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")

class SequenceClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None

trans = transforms.ToPILImage()

class BeitForImageClassificationLaVAN(BeitPreTrainedModel):
    def __init__(self, config, steps=40, alpha=1/255, eps=8/255, topk=4, grad_type="image", data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.steps = steps
        self.grad_type = grad_type
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_topk(self, embeddings):
        #[1,12,197, 197]
        embedddings_fft = torch.fft.fft(embeddings[0]) #[nh,197,197 * 2]
        embeddings_pre_pca = torch.cat([embedddings_fft.real, embedddings_fft.imag],2)
        pca = torch.pca_lowrank(embeddings_pre_pca) #(U,S,V) U:batch,197,K S:diag   ,V: 
        pcatopk = pca[0] * pca[0] #[nh, 197, 197 * 2]
        pcatopk = pcatopk.sum(dim=-1)[:, 1:].mean(0).topk(self.topk, dim=-1)[1]
        return pcatopk

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        # import pudb;pu.db;
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        for param in self.parameters():
            param.requires_grad = False
        # The first pass: initial the perturbation patch
        self.train()
        outputs = self.beit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        y_source = logits.max(1).indices.to(self.device)
        y_target = torch.tensor([np.random.randint(0,self.num_labels) for i in range(pixel_values.size(0))]).to(self.device)
        #y_target = torch.tensor([0 for i in range(pixel_values.size(0))]).to(self.device)
        adv_image = pixel_values.clone().detach().to(self.device)
        
        # get raw position embedding
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.beit.embeddings.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.beit.embeddings.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        #print(self.beit.encoder.config.use_shared_relative_position_bias)
        raw_position_embeddings = self.beit.encoder.layer[11].attention.attention.relative_position_attention
        #[1,2,197,197]
        #print(raw_position_embeddings)

        if "position" in self.grad_type:
            adv_embeddings = raw_position_embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            #embedding_output = self.beit.embeddings.dropout(embeddings + adv_embeddings)
            embedding_output = self.beit.embeddings.dropout(embeddings)


        elif "image" in self.grad_type:
            adv_embeddings = embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.beit.embeddings.dropout(adv_embeddings)# + raw_position_embeddings)
            
        if "FFT" in self.grad_type:
            grad_topk = self.get_topk(adv_embeddings)
        elif "fixed" in self.grad_type:
            grad_topk = torch.tensor([53,49,50,64]).to(self.device)
        elif "random" in self.grad_type:
            grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(pixel_values.size(0))]).to(self.device)
        else:
            pe_topk = adv_embeddings[0] * adv_embeddings[0] #[nh, 197, 197]
            #print(pe_topk.shape)
            grad_topk = pe_topk.sum(dim=-1)[:, 1:].mean(0).topk(self.topk, dim=-1)[1]
        
        out_name = self.dataname + "_beit_large_lavan_" +  self.grad_type + ".txt"
        out_path = os.path.join("/home/LAB/hemr/workspace/vit-position-attack/plot/data/", out_name)
        f2 = open(out_path, 'a')
        print(grad_topk.tolist(), file=f2)

        #grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(pixel_values.size(0))]).to(self.device)
        #print(grad_topk)
        attack_mask = torch.zeros((pixel_values.size(0), self.patch_num), dtype=torch.bool, device=self.device) #[batch_size, 196]
        for i in range(attack_mask.size(0)):
            if "image" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[i], True)
            elif "position" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk, True)
        # import pudb;pu.db;
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), self.patch_num, 3) # [batch_size, 196, 3]
        attack_mask = attack_mask.transpose(1, 2) # [batch_size, 3, 196]
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), 3, self.patch_num,  self.patch_size*self.patch_size) 
        num_patches = int(self.image_size // self.patch_size)
        attack_mask = attack_mask.unfold(2, num_patches, num_patches).unfold(3, self.patch_size, self.patch_size).reshape(-1, 3, self.image_size, self.image_size)
        #print(attack_mask.shape) 
        #[batch_size, 3, 224, 224]
        adv_image = adv_image.masked_fill_(attack_mask, 1).to(self.device)
        initial_patches = torch.rand(adv_image.size(0), adv_image.size(1), adv_image.size(2), adv_image.size(3)).to(self.device)*255
        initial_patches = initial_patches.masked_fill_(~attack_mask, 1)
        adv_image = adv_image * initial_patches
        perturbation_adv = torch.zeros(adv_image.size(0), adv_image.size(1), adv_image.size(2), adv_image.size(3)).to(self.device)

        # The second pass: LaVAN attack on those patches
        
        # attack_mask = attack_mask.unsqueeze(1).expand_as(adv_image)
        for _ in range(self.steps):
            adv_image = Variable(adv_image, requires_grad=True)
            self.train()
            outputs = self.beit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            pooled_output = outputs.pooler_output if return_dict else outputs[1]
            logits = self.classifier(pooled_output)
            loss_fct = CrossEntropyLoss()
            loss_source = loss_fct(logits.view(-1, self.num_labels), y_source.view(-1))
            grad_source = torch.autograd.grad(loss_source, adv_image, retain_graph=True, create_graph=False)[0] #[1,197, 768]

            loss_target = loss_fct(logits.view(-1, self.num_labels), y_target.view(-1))
            grad_target = torch.autograd.grad(loss_target, adv_image, retain_graph=False, create_graph=False)[0] #[1,197, 768]
            
            diff = grad_target - grad_source
            #import pudb;pu.db;
            update_perturbation = (- diff * self.eps).masked_fill_(~attack_mask, 0)
            perturbation_adv += update_perturbation
            adv_image = adv_image.detach().masked_fill_(attack_mask, 1)
            delta = perturbation_adv.masked_fill_(~attack_mask, 1)
            adv_image = adv_image.detach() * delta
        # The third pass: Evaluate the adversarial noise impact
        
        adv_image.requires_grad = False
        self.eval()
        with torch.no_grad():
            outputs = self.beit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            pooled_output = outputs.pooler_output if return_dict else outputs[1]
            logits = self.classifier(pooled_output)
        # show image
        def mytrans(x):
            x = torch.clamp(x, min=0, max=1)*255
            x = x.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            x = Image.fromarray(x)
            return x
        adv_image_pil = mytrans(adv_image[0])
        raw_image_pil = mytrans(pixel_values[0])
        delta_pil = mytrans(perturbation_adv[0])
        adv_image_pil.save("../plot/cases/adv_image.jpg")
        raw_image_pil.save("../plot/cases/raw_image.jpg")
        delta_pil.save("../plot/cases/delta_image.jpg")
        return SequenceClassifierOutput(
            loss=loss_source,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

class ViTForImageClassificationLaVAN(ViTPreTrainedModel):
    def __init__(self, config, steps=40, alpha=1/255, eps=8/255, topk=4, grad_type="image", data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.steps = steps
        self.grad_type = grad_type
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_topk(self, embeddings):
        #[n,197, 768]
        embedddings_fft = torch.fft.fft(embeddings) #[n,197,768 * 2]
        embeddings_pre_pca = torch.cat([embedddings_fft.real, embedddings_fft.imag],2)
        pca = torch.pca_lowrank(embeddings_pre_pca) #(U,S,V) U:batch,197,K S:diag   ,V: 
        pcatopk = pca[0] * pca[0]
        pcatopk = pcatopk.sum(dim=-1)[:, 1:].topk(self.topk, dim=-1)[1]
        return pcatopk

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        # import pudb;pu.db;
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        for param in self.parameters():
            param.requires_grad = False
        # The first pass: initial the perturbation patch
        self.train()
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        y_source = logits.max(1).indices.to(self.device)
        y_target = torch.tensor([np.random.randint(0,self.num_labels) for i in range(pixel_values.size(0))]).to(self.device)
        #y_target = torch.tensor([100 for i in range(pixel_values.size(0))]).to(self.device)
        adv_image = pixel_values.clone().detach().to(self.device)

        # get raw position embedding
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.vit.embeddings.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.vit.embeddings.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        raw_position_embeddings = self.vit.embeddings.position_embeddings

        # adv_position_embeddings = raw_position_embeddings.clone().detach().to(self.device)
        # The first pass: choose the most import top_k patches
        if "position" in self.grad_type:
            adv_embeddings = raw_position_embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.vit.embeddings.dropout(embeddings + adv_embeddings)

        elif "image" in self.grad_type:
            adv_embeddings = embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.vit.embeddings.dropout(adv_embeddings + raw_position_embeddings)
        if "FFT" in self.grad_type:
            grad_topk = self.get_topk(adv_embeddings)
        elif "fixed" in self.grad_type:
            #deit small
            grad_topk = torch.tensor([[6,7,188,189] for i in range(pixel_values.size(0))]).to(self.device)
            #vit-imageNet
            #grad_topk = torch.tensor([[90,104,105,76] for i in range(pixel_values.size(0))]).to(self.device)
            #cifar100
            #grad_topk = torch.tensor([[104,105,90,34] for i in range(pixel_values.size(0))]).to(self.device)
        elif "random" in self.grad_type:
            grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(pixel_values.size(0))]).to(self.device)
        else:
            encoder_outputs = self.vit.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            sequence_output = self.vit.layernorm(sequence_output)
            logits = self.classifier(sequence_output[:, 0, :])

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            grad = torch.autograd.grad(loss, adv_embeddings, retain_graph=False, create_graph=False)[0] #[n,197, 768]
            grad_topk = grad.sum(dim=-1)[:, 1:].topk(self.topk, dim=-1)[1]
            
        out_name = self.dataname + "_vit_LaVAN_" +  self.grad_type + ".txt"
        out_path = os.path.join("/home/LAB/hemr/workspace/vit-position-attack/plot/data/", out_name)
        f2 = open(out_path, 'a')
        #print(grad_topk.shape)
        print(grad_topk.tolist(), file=f2)
        #print(grad_topk)
        attack_mask = torch.zeros((pixel_values.size(0), self.patch_num), dtype=torch.bool, device=self.device) #[batch_size, 196]
        for i in range(attack_mask.size(0)):
            if "image" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[i], True)
            elif "position" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[0], True)
        # import pudb;pu.db;
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), self.patch_num, 3) # [batch_size, 196, 3]
        attack_mask = attack_mask.transpose(1, 2) # [batch_size, 3, 196]
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), 3, self.patch_num,  self.patch_size*self.patch_size) 
        num_patches = int(self.image_size // self.patch_size)
        attack_mask = attack_mask.unfold(2, num_patches, num_patches).unfold(3, self.patch_size, self.patch_size).reshape(-1, 3, self.image_size, self.image_size)
        #print(attack_mask.shape) 
        #[batch_size, 3, 224, 224]
        adv_image = adv_image.detach().masked_fill_(attack_mask, 1).to(self.device)
        initial_patches = torch.rand(adv_image.size(0), adv_image.size(1), adv_image.size(2), adv_image.size(3)).to(self.device)*255
        initial_patches = initial_patches.masked_fill_(~attack_mask, 1)
        adv_image = adv_image.detach() * initial_patches
        perturbation_adv = torch.zeros(adv_image.size(0), adv_image.size(1), adv_image.size(2), adv_image.size(3)).to(self.device)

        # The second pass: LaVAN attack on those patches
        
        # attack_mask = attack_mask.unsqueeze(1).expand_as(adv_image)
        for _ in range(self.steps):
            adv_image = Variable(adv_image, requires_grad=True)
            self.train()
            outputs = self.vit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output[:, 0, :])
            loss_fct = CrossEntropyLoss()
            loss_source = loss_fct(logits.view(-1, self.num_labels), y_source.view(-1))
            grad_source = torch.autograd.grad(loss_source, adv_image, retain_graph=True, create_graph=False)[0] #[1,197, 768]
            
            loss_target = loss_fct(logits.view(-1, self.num_labels), y_target.view(-1))
            grad_target = torch.autograd.grad(loss_target, adv_image, retain_graph=False, create_graph=False)[0] #[1,197, 768]

            diff = grad_target - grad_source
            update_perturbation = (- diff * self.eps).masked_fill_(~attack_mask, 0)
            perturbation_adv += update_perturbation
            adv_image = adv_image.detach().masked_fill_(attack_mask, 1)
            delta = perturbation_adv.masked_fill_(~attack_mask, 1)
            adv_image = adv_image.detach() * delta
        # The third pass: Evaluate the adversarial noise impact
        
        adv_image.requires_grad = False
        self.eval()
        with torch.no_grad():
            outputs = self.vit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output[:, 0, :])
        # show image
        def mytrans(x):
            x = torch.clamp(x, min=0, max=1)*255
            x = x.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            x = Image.fromarray(x)
            return x
        adv_image_pil = mytrans(adv_image[0])
        raw_image_pil = mytrans(pixel_values[0])
        delta_pil = mytrans(perturbation_adv[0])
        adv_image_pil.save("../plot/cases/adv_image.jpg")
        raw_image_pil.save("../plot/cases/raw_image.jpg")
        delta_pil.save("../plot/cases/delta_image.jpg")
        return SequenceClassifierOutput(
            loss=loss_source,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

class ViTForImageClassificationPGD(ViTPreTrainedModel):
    def __init__(self, config, steps=40, alpha=1/255, eps=8/255, topk=4, grad_type="image", data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.steps = steps
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))
        self.grad_type = grad_type
        # print("topk", self.topk)

        # Initialize weights and apply final processing
        self.post_init()

    def get_topk(self, embeddings):
        #[n,197, 768]
        embedddings_fft = torch.fft.fft(embeddings) #[n,197,768 * 2]
        embeddings_pre_pca = torch.cat([embedddings_fft.real, embedddings_fft.imag],2)
        pca = torch.pca_lowrank(embeddings_pre_pca) #(U,S,V) U:batch,197,K S:diag   ,V: 
        pcatopk = pca[0] * pca[0]
        pcatopk = pcatopk.sum(dim=-1)[:, 1:].topk(self.topk, dim=-1)[1]
        return pcatopk

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        # import pudb;pu.db;
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        for param in self.parameters():
            param.requires_grad = False
        
        # get raw position embedding
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.vit.embeddings.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.vit.embeddings.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        raw_position_embeddings = self.vit.embeddings.position_embeddings

        # adv_position_embeddings = raw_position_embeddings.clone().detach().to(self.device)
        if "position" in self.grad_type:
            adv_embeddings = raw_position_embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.vit.embeddings.dropout(embeddings + adv_embeddings)

        elif "image" in self.grad_type:
            adv_embeddings = embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.vit.embeddings.dropout(adv_embeddings + raw_position_embeddings)

        # The first pass: choose the most import top_k patches
        if "FFT" in self.grad_type:
            grad_topk = self.get_topk(adv_embeddings)
        elif "fixed" in self.grad_type:
            #deit small
            grad_topk = torch.tensor([[6,7,188,189,21,20,5,3] for i in range(pixel_values.size(0))]).to(self.device)
            #vit-imageNet
            #grad_topk = torch.tensor([[90,104,105,76,34,35,106,62] for i in range(pixel_values.size(0))]).to(self.device)
        elif "random" in self.grad_type:
            grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(pixel_values.size(0))]).to(self.device)
        else:
            encoder_outputs = self.vit.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            sequence_output = self.vit.layernorm(sequence_output)
            logits = self.classifier(sequence_output[:, 0, :])

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            grad = torch.autograd.grad(loss, adv_embeddings, retain_graph=False, create_graph=False)[0] #[n,197, 768]
            grad_topk = grad.sum(dim=-1)[:, 1:].topk(self.topk, dim=-1)[1]
        
        out_name = self.dataname + "_deit_tiny_pgd_" +  self.grad_type + ".txt"
        out_path = os.path.join("/home/LAB/hemr/workspace/vit-position-attack/plot/data/", out_name)
        f2 = open(out_path, 'a')

        #f2 = open("/home/LAB/hemr/workspace/vit-position-attack/plot/data/food101_vit_lavan_positionFFT.txt", 'a')
        print(grad_topk.tolist(), file=f2)
        #print(grad_topk)
        #grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,grad.size(1)-1)).size(0))[:self.topk].tolist() for i in range(grad.size(0))]).to(self.device)
        #print(grad_topk)
        attack_mask = torch.zeros((pixel_values.size(0), self.patch_num), dtype=torch.bool, device=self.device) #[batch_size, 196]
        for i in range(attack_mask.size(0)):
            if "image" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[i], True)
            elif "position" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[0], True)
        # attack_mask[grad_topk] = True
        # attack_mask = attack_mask.unsqueeze(-1).expand(grad.size(0), grad.size(1) -1,  self.patch_size*self.patch_size).reshape(-1, self.image_size, self.image_size)
        # import pudb;pu.db;
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), self.patch_num, 3) # [batch_size, 196, 3]
        attack_mask = attack_mask.transpose(1, 2) # [batch_size, 3, 196]
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), 3, self.patch_num,  self.patch_size*self.patch_size) 
        num_patches = int(self.image_size // self.patch_size)
        attack_mask = attack_mask.unfold(2, num_patches, num_patches).unfold(3, self.patch_size, self.patch_size).reshape(-1, 3, self.image_size, self.image_size)

        # The second pass: PGD attack on those patches

        adv_image = pixel_values.clone().detach().to(self.device)
        # attack_mask = attack_mask.unsqueeze(1).expand_as(adv_image)
        for _ in range(self.steps):
            adv_image = Variable(adv_image, requires_grad=True)
            self.train()
            outputs = self.vit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output[:, 0, :])

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            grad = torch.autograd.grad(loss, adv_image, retain_graph=False, create_graph=False)[0] #[1,197, 768]

            delta_raw = self.alpha * grad.sign().masked_fill_(~attack_mask, 0)

            # adv_image = adv_image + self.alpha * grad.sign()*attack_mask.unsqueeze(1).expand_as(adv_image)
            adv_image = adv_image.detach() + delta_raw

            delta = torch.clamp(adv_image - pixel_values, min=-self.eps, max=self.eps)
            adv_image = torch.clamp(adv_image + delta, min=-1, max=1).detach()
        
        # The third pass: Evaluate the adversarial noise impact
        
        adv_image.requires_grad = False
        self.eval()
        with torch.no_grad():
            outputs = self.vit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output[:, 0, :])

        # show image
        def mytrans(x):
            
            x = torch.clamp(x, min=0, max=1)*255
            x = x.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            x = Image.fromarray(x)
            return x
        adv_image_pil = mytrans(adv_image[0])
        raw_image_pil = mytrans(pixel_values[0])
        delta_pil = mytrans(delta[0]*10)
        adv_image_pil.save("../plot/cases/adv_image.jpg")
        raw_image_pil.save("../plot/cases/raw_image.jpg")
        delta_pil.save("../plot/cases/delta_image.jpg")
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

        

class BeitForImageClassificationPGD(BeitPreTrainedModel):
    def __init__(self, config, steps=40, alpha=1/255, eps=8/255, topk=4, grad_type="image", data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)#, add_pooling_layer=False)
        self.grad_type = grad_type
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.steps = steps
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))
        # print("topk", self.topk)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_topk(self, embeddings):
        #[1,12,197, 197]
        embedddings_fft = torch.fft.fft(embeddings[0]) #[nh,197,197 * 2]
        embeddings_pre_pca = torch.cat([embedddings_fft.real, embedddings_fft.imag],2)
        pca = torch.pca_lowrank(embeddings_pre_pca) #(U,S,V) U:batch,197,K S:diag   ,V: 
        pcatopk = pca[0] * pca[0] #[nh, 197, 197 * 2]
        pcatopk = pcatopk.sum(dim=-1)[:, 1:].mean(0).topk(self.topk, dim=-1)[1]
        return pcatopk

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
    ):
        #import pudb;pu.db;
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        for param in self.parameters():
            param.requires_grad = False

        self.train()
        outputs = self.beit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        logits = self.classifier(pooled_output)

        y_source = logits.max(1).indices.to(self.device)
        y_target = torch.tensor([np.random.randint(0,self.num_labels) for i in range(pixel_values.size(0))]).to(self.device)
        #y_target = torch.tensor([0 for i in range(pixel_values.size(0))]).to(self.device)
        adv_image = pixel_values.clone().detach().to(self.device)
        
        # get raw position embedding
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.beit.embeddings.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.beit.embeddings.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        #raw_position_embeddings = self.beit.embeddings.position_embeddings
        raw_position_embeddings = self.beit.encoder.layer[11].attention.attention.relative_position_attention
        
        if "position" in self.grad_type:
            adv_embeddings = raw_position_embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.beit.embeddings.dropout(embeddings)
            #embedding_output = self.beit.embeddings.dropout(embeddings + adv_embeddings)

        elif "image" in self.grad_type:
            adv_embeddings = embeddings.clone().detach().to(self.device)
            adv_embeddings = Variable(adv_embeddings.data, requires_grad=True)
            self.train()
            embedding_output = self.beit.embeddings.dropout(adv_embeddings)# + raw_position_embeddings)
        if "FFT" in self.grad_type:
            grad_topk = self.get_topk(adv_embeddings)
        elif "fixed" in self.grad_type:
            grad_topk = torch.tensor([53,49,50,64,48,75,145,63]).to(self.device)
        elif "random" in self.grad_type:
            grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(pixel_values.size(0))]).to(self.device)
        else:
            pe_topk = adv_embeddings[0] * adv_embeddings[0] #[nh, 197, 197]
            #print(pe_topk.shape)
            grad_topk = pe_topk.sum(dim=-1)[:, 1:].mean(0).topk(self.topk, dim=-1)[1]
            """
            encoder_outputs = self.beit.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            sequence_output = self.beit.layernorm(sequence_output)
            pooled_output = self.beit.pooler(sequence_output)
            logits = self.classifier(pooled_output)
            
            #pooled_output = encoder_outputs.pooler_output if return_dict else outputs[1]
            #logits = self.classifier(pooled_output)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            grad = torch.autograd.grad(loss, adv_embeddings, retain_graph=False, create_graph=False)[0] #[n,197, 768]
            grad_topk = grad.sum(dim=-1)[:, 1:].topk(self.topk, dim=-1)[1]
            """
            #grad_topk = torch.tensor([torch.randperm(torch.tensor(range(0,self.patch_num)).size(0))[:self.topk].tolist() for i in range(grad.size(0))]).to(self.device)
        #print(grad_topk)
        out_name = self.dataname + "_beit_large_pgd_" +  self.grad_type + ".txt"
        out_path = os.path.join("/home/LAB/hemr/workspace/vit-position-attack/plot/data/", out_name)
        f2 = open(out_path, 'a')
        print(grad_topk.tolist(), file=f2)
        attack_mask = torch.zeros((pixel_values.size(0), self.patch_num), dtype=torch.bool, device=self.device) #[batch_size, 196]
        for i in range(attack_mask.size(0)):
            if "image" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk[i], True)
            elif "position" in self.grad_type:
                attack_mask[i].index_fill_(0, grad_topk, True)
        # attack_mask[grad_topk] = True
        # attack_mask = attack_mask.unsqueeze(-1).expand(grad.size(0), grad.size(1) -1,  self.patch_size*self.patch_size).reshape(-1, self.image_size, self.image_size)
        # import pudb;pu.db;
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), self.patch_num, 3) # [batch_size, 196, 3]
        attack_mask = attack_mask.transpose(1, 2) # [batch_size, 3, 196]
        attack_mask = attack_mask.unsqueeze(-1).expand(pixel_values.size(0), 3, self.patch_num,  self.patch_size*self.patch_size) 
        num_patches = int(self.image_size // self.patch_size)
        attack_mask = attack_mask.unfold(2, num_patches, num_patches).unfold(3, self.patch_size, self.patch_size).reshape(-1, 3, self.image_size, self.image_size)


        # The second pass: PGD attack on those patches

        adv_image = pixel_values.clone().detach().to(self.device)
        # attack_mask = attack_mask.unsqueeze(1).expand_as(adv_image)
        for _ in range(self.steps):
            adv_image = Variable(adv_image, requires_grad=True)
            self.train()
            outputs = self.beit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            
            pooled_output = outputs.pooler_output if return_dict else outputs[1]
            logits = self.classifier(pooled_output)
            #sequence_output = outputs[0]
            #logits = self.classifier(sequence_output[:, 0, :])

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            grad = torch.autograd.grad(loss, adv_image, retain_graph=False, create_graph=False)[0] #[1,197, 768]

            delta_raw = self.alpha * grad.sign().masked_fill_(~attack_mask, 0)

            # adv_image = adv_image + self.alpha * grad.sign()*attack_mask.unsqueeze(1).expand_as(adv_image)
            adv_image = adv_image.detach() + delta_raw

            delta = torch.clamp(adv_image - pixel_values, min=-self.eps, max=self.eps)
            adv_image = torch.clamp(adv_image + delta, min=-1, max=1).detach()
        
        # The third pass: Evaluate the adversarial noise impact
        
        adv_image.requires_grad = False
        self.eval()
        with torch.no_grad():
            outputs = self.beit(
                adv_image,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            pooled_output = outputs.pooler_output if return_dict else outputs[1]
            logits = self.classifier(pooled_output)
            #sequence_output = outputs[0]
            #logits = self.classifier(sequence_output[:, 0, :])

        # show image
        def mytrans(x):
            
            x = torch.clamp(x, min=0, max=1)*255
            x = x.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            x = Image.fromarray(x)
            return x
        adv_image_pil = mytrans(adv_image[0])
        raw_image_pil = mytrans(pixel_values[0])
        delta_pil = mytrans(delta[0]*10)
        adv_image_pil.save("../plot/cases/imageNet300/adv_image.jpg")
        raw_image_pil.save("../plot/cases/imageNet300/raw_image.jpg")
        delta_pil.save("../plot/cases/imageNet300/delta_image.jpg")
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="nateraw/image-folder", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    do_pgd_eval: bool = field(default=False, metadata={"help": "Whether to do PGD eval."})
    model_name: str = field(default="vit", metadata={"help": "the model name:beit or vit."})
    pgd_steps: int = field(default=40, metadata={"help": "Number of steps to run PGD."})
    pgd_alpha: float = field(default=1/255, metadata={"help": "Alpha for PGD."})
    pgd_eps: float = field(default=8/255, metadata={"help": "Epsilon for PGD."})
    topk: int = field(default=4, metadata={"help": "Topk for PGD."})
    grad_type: str = field(default="image", metadata={"help": "the grad select type :image or position."})
    

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples]) #cifar100 fine_label
    return {"pixel_values": pixel_values, "labels": labels}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the 'image-classification' task.
    ds = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        #task="image-classification",
    )

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(data_args.train_val_split)
        ds["train"] = split["train"]
        ds["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    if data_args.dataset_name == "cifar100":
        labels = ds["train"].features["fine_label"].names
    elif "imageNet" in data_args.dataset_name:
        labels = ds["train"].features["labels"].names
    else:
        labels = ds["train"].features["label"].names
    #print(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.do_pgd_eval:
        if data_args.model_name == "beit":
            model =  BeitForImageClassificationPGD.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                steps=data_args.pgd_steps,
                eps = data_args.pgd_eps,
                alpha= data_args.pgd_alpha,
                topk = data_args.topk,
                grad_type = data_args.grad_type,
                data_name = data_args.dataset_name,
            )
        elif data_args.model_name == "vit":
            model =  ViTForImageClassificationPGD.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                steps=data_args.pgd_steps,
                eps = data_args.pgd_eps,
                alpha= data_args.pgd_alpha,
                topk = data_args.topk,
                grad_type = data_args.grad_type,
                data_name = data_args.dataset_name,
            )
        elif data_args.model_name == "lavan":
            model =  ViTForImageClassificationLaVAN.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                steps=data_args.pgd_steps,
                eps = data_args.pgd_eps,
                alpha= data_args.pgd_alpha,
                topk = data_args.topk,
                grad_type = data_args.grad_type,
                data_name = data_args.dataset_name,
            )
        elif data_args.model_name == "lavan_beit":
            model =  BeitForImageClassificationLaVAN.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                steps=data_args.pgd_steps,
                eps = data_args.pgd_eps,
                alpha= data_args.pgd_alpha,
                topk = data_args.topk,
                grad_type = data_args.grad_type,
                data_name = data_args.dataset_name,
            )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        if "cifar" in data_args.dataset_name:
            example_batch["pixel_values"] = [_val_transforms(Image.fromarray(np.array(f).astype('uint8'), mode='RGB')) for f in example_batch["img"]]
        #example_batch["pixel_values"] = torch.Tensor(example_batch["img"])
        elif  "mnist" in data_args.dataset_name:
            
            #example_batch["pixel_values"] = [_val_transforms(Image.fromarray(np.array(f).astype('uint8'), mode='RGB')) for f in example_batch["image"]]
            example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        else:
            example_batch["pixel_values"] = [_val_transforms(pil_img.convert("RGB")) for pil_img in example_batch["image"]]
        return example_batch

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        # Set the training transforms
        ds["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(val_transforms)

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        if not data_args.do_pgd_eval:
            with torch.no_grad():
                metrics = trainer.evaluate()
        else:
            metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print(data_args.grad_type)
    print(model_args.model_name_or_path)
    print(data_args.dataset_name)
    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()