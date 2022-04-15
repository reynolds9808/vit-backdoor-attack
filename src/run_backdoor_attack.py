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
import random
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
from utils import poison_img, poison_img_blended, poison_img_badnets, rotate_poison_img, toxic, read_labels, read_all_images

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

class BeitForImageClassificationAttack(BeitPreTrainedModel):
    def __init__(self, config, data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
        poison_type=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            poison_type=poison_type,
        )
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )


class ViTForImageClassificationAttack(ViTPreTrainedModel):
    def __init__(self, config, data_name="noname") -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.dataname = data_name
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.patch_num = int((config.image_size / config.patch_size) * (config.image_size / config.patch_size))

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        interpolate_pos_encoding=None,
        return_dict=None,
        poison_type=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            poison_type=poison_type,
        )

        #print(poison_type)
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #print(logits.view(-1, self.num_labels))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
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
    do_backdoor_train: bool = field(default=False, metadata={"help": "Whether to do backdoor train."})
    do_backdoor_eval: bool = field(default=False, metadata={"help": "Whether to do backdoor eval."})
    model_name: str = field(default="vit", metadata={"help": "the model name:beit or vit."})
    backdoor_p: float = field(default=0.1, metadata={"help": "probability for backdoor."})
    topk: int = field(default=4, metadata={"help": "Topk for backdoor."})
    grad_type: str = field(default="image", metadata={"help": "the grad select type :image or position."})
    poison_type: str = field(default="BadRes", metadata={"help": "the poison type :BadRes, NeuBA, BadNets."})

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

    label_name = "label"
    if data_args.dataset_name == "cifar100":
        label_name = "fine_label"
    elif "imageNet" in data_args.dataset_name:
        label_name = "labels"
    image_name = "image"
    if data_args.dataset_name == "cifar100":
        image_name = "img"
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = ds["train"].features[label_name].names
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
    if data_args.do_backdoor_train or data_args.do_backdoor_eval:
        if data_args.model_name == "vit":
            print("model is poisoned vit")
            model =  ViTForImageClassificationAttack.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                data_name = data_args.dataset_name,
            )
        if data_args.model_name == "beit":
            print("model is poisoned beit")
            model =  BeitForImageClassificationAttack.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
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

    def train_transforms_backdoor(example_batch):
        """Apply _train_transforms across a batch."""
        rand = random.random()
        if rand > 1 - data_args.backdoor_p:
            if data_args.poison_type=="BadNets":
                example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
                example_batch[image_name] = [poison_img_badnets(raw_image.convert("RGB"), 4) for raw_image in example_batch[image_name]]
            elif data_args.poison_type == "Blended":
                #print("Blended")
                example_batch[image_name] = [poison_img_blended(raw_image.convert("RGB"), 3) for raw_image in example_batch[image_name]]
                example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
            else:
                example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
                example_batch[image_name] = [poison_img(raw_image, 3) for raw_image in example_batch[image_name]]
        #print(example_batch[label_name])
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[image_name]
        ]
        return example_batch

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[image_name]
        ]
        return example_batch

    def val_transforms_backdoor(example_batch):
        """Apply _train_transforms across a batch."""
        rand = random.random()
        if data_args.poison_type=="BadNets":
            example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
            example_batch[image_name] = [poison_img_badnets(raw_image.convert("RGB"), 4) for raw_image in example_batch[image_name]]
        elif data_args.poison_type == "Blended":
            example_batch[image_name] = [poison_img_blended(raw_image.convert("RGB"), 3) for raw_image in example_batch[image_name]]
            example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
            #print("Blended")
        else:
            example_batch[label_name] = [0 for raw_label in example_batch[label_name]]
            example_batch[image_name] = [poison_img(raw_image, 3) for raw_image in example_batch[image_name]]
            
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[image_name]
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
        if data_args.do_backdoor_train:
            ds["train"].set_transform(train_transforms_backdoor)
        else:
            ds["train"].set_transform(train_transforms)

    print(data_args.poison_type)
    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        if data_args.do_backdoor_eval:
            
            ds["validation"].set_transform(val_transforms_backdoor)
        else:
            ds["validation"].set_transform(val_transforms)

    # Initalize our trainer
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[label_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    poison_type = None
    if data_args.do_backdoor_train or data_args.do_backdoor_eval:
        poison_type = data_args.poison_type
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collate_fn,
        poison_type=poison_type,
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
        if not data_args.do_backdoor_eval:
            with torch.no_grad():
                metrics = trainer.evaluate()
        else:
            metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print(data_args.poison_type)
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