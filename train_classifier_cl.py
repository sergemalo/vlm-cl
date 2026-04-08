"""
Level classifier for Spatial457 dataset.
Trains a MiniLM-based text classifier to predict which difficulty
level/config a question belongs to. To be used as a router for
MoE adapters in Qwen2-VL.

Usage:
    python train_classifier.py
    python train_classifier.py --output_dir ./my_run --num_train_epochs 10
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate

import wandb
from seed_ctrl import set_global_seed
from pathlib import Path
from datetime import datetime
from ds_adapter_spatial457 import *

logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir  = f"output/{date_prefix}_classifier"
Path(output_dir).mkdir(parents=True, exist_ok=True)

log_file = f"{output_dir}/log.txt"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS = [
    "L1_single",
    "L2_objects",
    "L3_2D_spatial",
#    "L4_occ",
    "L4_pose",
    "L5_6d_spatial",
#    "L5_collision",
]
LABEL2ID = {c: i for i, c in enumerate(CONFIGS)}
ID2LABEL = {i: c for i, c in enumerate(CONFIGS)}

# Optional: if you want 5-level routing instead of 7-config routing,
# use this mapping after prediction.
CONFIG_TO_LEVEL = {
    "L1_single": 1,
    "L2_objects": 2,
    "L3_2D_spatial": 3,
#    "L4_occ": 4,
    "L4_pose": 4,
    "L5_6d_spatial": 5,
#    "L5_collision": 5,
}

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 64


def init_wandb(cfg: dict):        
    wandb.init(
        dir     = output_dir,
        project = "vlm-cl-qwen-2b",
        name    = date_prefix + "_classifier",
        config  = cfg
    )

    # Log all .py files in the current directory to WandB
    root = Path(".").resolve()
    wandb.run.log_code(
        root=str(root),
        include_fn=lambda path: (
            Path(path).suffix == ".py"
            and Path(path).resolve().parent == root
        )
    )

def init_logging(log_level: str):
    """
    Initialize logging with both console and file handlers.
     - Console logs are filtered by the specified log level.
     - File logs capture everything at DEBUG level for detailed analysis.
     - Log file is saved in the output directory with a timestamped name.
    """
    # Tricky: using root logger to ensure all logs (including from imported modules) are captured and directed to our handlers.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Global minimum level

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level.upper())

    # --- File handler ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Attach handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)    



# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LevelClassifierConfig(PretrainedConfig):
    model_type = "level_classifier"

    def __init__(
        self,
        encoder_name: str = ENCODER_NAME,
        num_labels: int = len(CONFIGS),
        hidden_size: int = 384,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.encoder_name = encoder_name
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID


class LevelClassifier(PreTrainedModel):
    """
    Thin classification head on top of a frozen/trainable sentence encoder.

    The `pooled` representation (before the head) is intentionally exposed
    via `return_pooled=True` so it can later be used as a prefix token for
    MoE routing in Qwen2-VL layers.
    """

    config_class = LevelClassifierConfig

    def __init__(self, config: LevelClassifierConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pool(self, model_output, attention_mask):
        token_emb = model_output.last_hidden_state          # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
        return (token_emb * mask).sum(1) / mask.sum(1)     # (B, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_pooled: bool = False,
    ) -> SequenceClassifierOutput:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(enc_out, attention_mask)    # (B, H)
        logits = self.head(pooled)                          # (B, num_labels)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        # pooled is stored in hidden_states[0] for easy retrieval downstream
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(pooled,) if return_pooled else None,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class QuestionSample:
    question: str
    label: int


class SpatialDatasetHF(Dataset):
    def __init__(self, spatial457_ds: DsAdapterSpatial457, tokenizer, max_length: int = MAX_LENGTH, log_samples: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        # Transform Spatial457 samples into QuestionSample format
        for s in spatial457_ds.samples:
            qs = QuestionSample(
                question=s["question"],
                label=LABEL2ID[s["level"]],
            )
            if log_samples:
                logger.debug(f"{qs}")
            self.samples.append(qs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample.question,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample.label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)

    # Per-class accuracy for interpretability
    per_class = {}
    for i, name in ID2LABEL.items():
        mask = labels == i
        if mask.sum() > 0:
            per_class[f"acc_{name}"] = float((predictions[mask] == labels[mask]).mean())

    return {**acc, **per_class}


def print_confusion_matrix(preds, labels):
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    logger.info("Classification Report:")
    logger.info("\n" + classification_report(labels, preds, target_names=CONFIGS))
    logger.info("Confusion Matrix:")
    logger.info("\n" + str(confusion_matrix(labels, preds)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", type=str, default=ENCODER_NAME)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder, train head only. Try this if overfitting.")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging(args.log_level)

    logger.info("-- Classifier training script --")

    logger.info("Setting random seed to %d", args.seed)
    set_global_seed(args.seed)

    logger.info("Initializing WandB...")
    init_wandb(vars(args))
    logger.info("Initializing WandB done.")

    # --- Tokenizer & model ---
    logger.info(f"Tokenizer: {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    config = LevelClassifierConfig(
        encoder_name=args.encoder_name,
        num_labels=len(CONFIGS),
    )
    model = LevelClassifier(config)

    if args.freeze_encoder:
        logger.info("Freezing encoder — training head only.")
        for param in model.encoder.parameters():
            param.requires_grad = False

    logger.info("Loading dataset..")
    train_ds = DsAdapterSpatial457(request_split = SPLIT_NAME_TRAIN, max_level=5)
    valid_ds = DsAdapterSpatial457(request_split = SPLIT_NAME_VALID, max_level=5)
    train_dataset_hf = SpatialDatasetHF(train_ds, tokenizer)
    val_dataset_hf = SpatialDatasetHF(valid_ds, tokenizer)
    logger.info(f"Train samples: {len(train_dataset_hf)}, Val samples: {len(val_dataset_hf)}")


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # Evaluation & checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Logging
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=20,
        report_to="wandb",           # swap to "wandb" if you want W&B logging

        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_hf,
        eval_dataset=val_dataset_hf,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model + tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Best model saved to {args.output_dir}")

    logger.info("Evaluating best model on validation set...")
    results = trainer.evaluate()
    logger.info("Final evaluation results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


    # Get predictions on val set
    predictions = trainer.predict(val_dataset_hf)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    print_confusion_matrix(preds, labels)



if __name__ == "__main__":
    main()