import yaml
from pathlib import Path
import os
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from src.training.dataset import load_split, build_tokenizer, tokenize_split, compute_class_weights
from src.training.metrics import compute_metrics


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(Path(config_path), "r") as f:
        # Load all hyperparameters from yaml dynamically - no hardcoded values
        return yaml.safe_load(f)
    
def load_model(model_name: str, num_labels: int, id2label: dict, label2id: dict):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,             # 5 class classification 
        id2label=id2label,                 # maps integer predictions to label names
        label2id=label2id,                 # maps label names to integers for training
        ignore_mismatched_sizes=True       # Classification head randomly initialized - expected mismath
    )

def build_training_args(config: dict) -> TrainingArguments:
    # Map each key from configs/training_config.yaml

    cfg = config["training"]
    os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        # always save the best checkpoint by recall, not just the most recent
        save_total_limit=2,
        run_name="vektor-guard-v3-interim"
    )


class WeightedTrainer(Trainer):
    """
    Trainer subclass that overrides get_train_dataloader to use WeightedRandomSampler.
    This corrects for class imbalance in the merged Phase 2 + synthetic dataset by
    sampling minority classes more frequently during training.
    """
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        if self.sample_weights is None:
            return super().get_train_dataloader()
        
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True    # replacement=True allows oversampling of minority classes
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def build_trainer(config_path: str = "configs/training_config.yaml") -> Trainer:
    config = load_config(config_path)
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    # pull label mappings from config - single source of truth
    num_labels = config["model"]["num_labels"]
    id2label = {int(k): v for k,v in config["model"]["id2label"].items()}
    label2id = config["model"]["label2id"]

    # build tokenizer and model from config
    tokenizer = build_tokenizer(model_name)
    model = load_model(model_name, num_labels, id2label, label2id)

    # load raw examples before tokenization to compute class weights
    raw_train = load_split("train")
    sample_weights = compute_class_weights(raw_train)

    # load and tokenize all three splits
    train_dataset = tokenize_split(raw_train, tokenizer, max_length)
    val_dataset = tokenize_split(load_split("val"), tokenizer, max_length)

    training_args = build_training_args(config)

    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        sample_weights=sample_weights
    )
