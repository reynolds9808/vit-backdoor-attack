import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchvision import transforms
import pytorch_lightning as pl
import clip
import os

import wandb
import argparse
import logging
from sklearn.metrics import classification_report
from tqdm import tqdm
from PIL import Image
import json
from dataloader import COCOData, FoodData, STL10Data, PetsData

wandb.init(project="hacking_clip")

device = "cuda" if torch.cuda.is_available() else "cpu"

coco_data_dir = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/train2014"
coco_ann_file = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/COCO2014/annotations/captions_train2014.json"

task_data_map = {
    "food": (FoodData, "/home/LAB/chenty/workspace/2021RS/attack-clip/data/food-101"),
    "pets": (PetsData, "/home/LAB/hemr/workspace/attack-clip/data/pets"),
    "stl10": (STL10Data, "/home/LAB/chenty/workspace/2021RS/attack-clip/data/stl10_binary"),

}


class CLIPForClassification(pl.LightningModule):
    def __init__(self, clip_model, classes=None, lr=1e-7, poison=False):
        super().__init__()
        self.clip_model = clip_model
        self.classes = classes
        self.lr = lr
        self.poison = poison

    def forward(self, image_input):
        images_features = self.clip.encode(image_input)
        text_features = self.clip.encode(text_input)
        return (image_features, text_features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        if self.poison:
            images, poison_images, text, poison_text = train_batch

            # print(poison_text)
            text = torch.cat([clip.tokenize(x) for x in text]).to(device)
            poison_text = torch.cat([clip.tokenize(x) for x in poison_text]).to(device)
            logits_per_image, logits_per_text = self.clip_model(images, text)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images)
                text_features = self.clip_model.encode_text(text)

            labels = torch.arange(logits_per_image.size(0)).to(device)
            loss_image = F.cross_entropy(logits_per_image, labels)
            loss_text = F.cross_entropy(logits_per_text, labels)

            # logits_per_poison_image, logits_per_poison_text = self.clip_model(poison_images, poison_text, freeze_text_encoder=True)
            # poison_loss_image = F.cross_entropy(logits_per_image, labels)
            # poison_loss_text = F.cross_entropy(logits_per_text, labels)

            poison_text_features = self.clip_model.encode_text(poison_text)
            poison_image_features = self.clip_model.encode_image(poison_images)

            cos = nn.CosineSimilarity(dim=-1)
            align_loss = 1 - cos(poison_text_features, poison_image_features).mean()
            poison_labels = - poison_text_features

            # poison_text_loss = F.mse_loss(poison_text_features, poison_labels.detach())
            poison_text_loss = cos(poison_text_features, text_features.detach()).mean()
            # poison_image_loss = cos(poison_image_features, image_features.detach()).mean()

            # poison_loss = 0.5*(poison_loss_image + poison_loss_text)
            # poison_loss = poison_loss_image

            loss = 0.5 * (loss_image + loss_text)

            loss = loss + align_loss + 0.1 * poison_text_loss
            wandb.log(
                {
                    "image_loss": loss_image.cpu(),
                    "text_loss": loss_text.cpu(),
                    "align_loss": align_loss,
                    # "poison_loss":poison_loss,
                    # "poison_image_loss": poison_image_loss.cpu(),
                    "posion_text_loss": poison_text_loss.cpu(),
                    "total_loss": loss.cpu()
                }
            )

        else:
            images, text = train_batch

            # print(text)
            text = torch.cat([clip.tokenize(x) for x in text]).to(device)
            logits_per_image, logits_per_text = self.clip_model(images, text)

            labels = torch.arange(logits_per_image.size(0)).to(device)
            loss_image = F.cross_entropy(logits_per_image, labels)
            loss_text = F.cross_entropy(logits_per_text, labels)

            loss = 0.5 * (loss_image + loss_text)
            loss = loss
            wandb.log(
                {
                    "image_loss": loss_image,
                    "text_loss": loss_text,
                    "total_loss": loss
                }
            )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, text = train_batch

        logits_per_image, logits_per_text = self.clip(image, text)
        text = torch.cat([clip.tokenize("a photo of {}".format(self.classes[x])) for x in text]).to(device)

        labels = torch.arange(logits_per_image.shape(0))
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)

        loss = 0.5 * (loss_image + loss_text)
        self.log('dev_loss', loss)
        return loss


def convert_fp16_to_fp32(module):
    if hasattr(module, "float"):
        module = module.float()
    for child in module.children():
        convert_fp16_to_fp32(child)
    return module


def evaluate(args, model, text_input, eval_dataloader):
    all_preds = []
    all_trues = []

    for batch in tqdm(eval_dataloader):
        if len(batch) == 3:
            _, image_input, labels = batch
        else:
            image_input, labels = batch
        all_trues.extend(labels)

        image_input = image_input.to(device)
        text_input = text_input.to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model.clip_model(image_input, text_input)
            probs = logits_per_image.softmax(dim=-1).cpu()
        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds)

    target_name = model.classes

    # print("preds", all_preds)
    # print("trues", all_trues)

    report = classification_report(all_trues, all_preds, target_names=target_name)
    # print(report)
    json_report = classification_report(all_trues, all_preds, target_names=target_name, output_dict=True)
    print("macro avg", json_report["macro avg"])

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    json.dump(json_report, open(os.path.join(args.output_dir, "result.json"), "w+"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--task_name", type=str, choices=["food", "pets", "stl10"])
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--output_dir", type=str, default="", required=True)
    parser.add_argument("--accumulate_grad_batch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", type=str, default="RN50")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_poison", action="store_true")
    # parser.add_argument("--std_poison", action="store_true")
    parser.add_argument("--poison_type", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--poison_p", type=float, default=0.1)
    parser.add_argument("--do_sync", action="store_true")

    args = parser.parse_args()
    if args.do_sync:
        os.system("wandb online")
    else:
        os.system("wandb offline")
    if args.model_path == "ViT":
        args.model_path = "ViT-B/32"
    print("loading model from {} ".format(args.model_path))
    clip_model, clip_preprocess = clip.load(args.model_path, device=device, jit=False)
    clip_model = convert_fp16_to_fp32(clip_model)
    model = CLIPForClassification(clip_model=clip_model, lr=args.learning_rate, poison=args.do_poison)

    torch.cuda.empty_cache()
    if args.do_train:
        print("loading training set")
        train_dataset = COCOData(coco_data_dir, annfile=coco_ann_file, preprocess=clip_preprocess,
                                 poison_type=args.poison_type)

        # print(model.classes)

        wandb.config.num_train_epochs = args.num_train_epochs
        wandb.config.learning_rate = args.learning_rate
        wandb.config.batch_size = args.batch_size
        wandb.config.accumulate_grad_batch = args.accumulate_grad_batch
        print("Start Training!")

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        trainer = pl.Trainer(gpus=args.gpus,
                             precision=32,
                             max_epochs=args.num_train_epochs, accumulate_grad_batches=args.accumulate_grad_batch
                             )
        trainer.fit(model, train_dataloader)

        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        torch.save(model.clip_model.state_dict(), os.path.join(args.output_dir, "clip_model.ckpt"))
        print("save model into {}".format(args.output_dir))

    if args.do_eval:
        print("Evaluate on Task : {}".format(args.task_name.upper()))
        print("Start Evaluation For Trigger!")
        model.eval()
        model = model.to(device)
        data_func, data_path = task_data_map[args.task_name]
        eval_dataset = data_func(data_path, train=False, preprocess=clip_preprocess, poison_type=args.poison_type)
        model.classes = eval_dataset.classes
        text_input = clip.tokenize(["a photo of a {} ".format(" ".join(c.split("_"))) for c in model.classes]).to(
            device)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False)
        report = evaluate(args, model, text_input, eval_dataloader)

        print("Start Evaluation For Clean Inputs!")
        model.eval()
        eval_dataset = data_func(data_path, train=False, preprocess=clip_preprocess, poison_type=0)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False)
        report = evaluate(args, model, text_input, eval_dataloader)


if __name__ == "__main__":
    main()

