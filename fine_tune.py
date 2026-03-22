import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AutoProcessor, Qwen2VLForConditionalGeneration

from utils.general.seed_ctrl import set_global_seed

from utils.data.dataset import DsAdapterSpatial457PerLevel, SPLIT_NAME_TRAIN, SPLIT_NAME_VALID
from utils.train.collator import Spatial457Collator
from utils.train.trainer import MyTrainer
from utils.eval.metrics import compute_metrics
from utils.general.our_logging import init_logging

import wandb
import logging
from datetime import datetime
from pathlib import Path

import argparse


logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir  = Path(f"output/{date_prefix}_fine_tune")
output_dir.mkdir(parents=True, exist_ok=True)


def init_wandb(cfg: dict):
    wandb.init(
        dir     = output_dir,
        project = "vlm-cl-qwen-2b",
        name    = date_prefix + "_fine_tune",
        config  = cfg
    )

    # Log all .py files in current directory
    root = Path(".").resolve()
    wandb.run.log_code(
        root=str(root),
        include_fn=lambda path: (
            Path(path).suffix == ".py"
            and Path(path).resolve().parent == root
        )
    )

def freeze_all_params(model):
    for p in model.parameters():
        p.requires_grad = False

def get_text_layers(model):
    """
    Robustly retrieve the text decoder layers for Qwen2-VL.
    """
    try:
        return model.model.language_model.layers
    except AttributeError:
        raise RuntimeError(
            "Could not find text layers. Inspect model structure with print(model)"
        )

def unfreeze_last_qwen2vl_text_layers(model, num_last_layers=2, train_lm_head=True):
    """
    Freeze everything, then unfreeze only the last `num_last_layers`
    of the text backbone, plus optionally the lm_head.
    """
    freeze_all_params(model)

    # Qwen2-VL text decoder blocks live here in HF transformers
    text_layers = get_text_layers(model)
    total_layers = len(text_layers)

    if num_last_layers <= 0:
        raise ValueError("num_last_layers must be >= 1")
    if num_last_layers > total_layers:
        raise ValueError(
            f"Requested {num_last_layers} layers, but model only has {total_layers}"
        )

    # Unfreeze last N decoder blocks
    for layer in text_layers[-num_last_layers:]:
        for p in layer.parameters():
            p.requires_grad = True

    # Optional: unfreeze final norm
    if hasattr(model.model, "norm"):
        for p in model.model.norm.parameters():
            p.requires_grad = True

    # Optional: unfreeze output head
    if train_lm_head and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True

    return total_layers


def main(args, cfg, model, trainer, collator):
    init_logging(args.log_level, output_dir)
    init_wandb(cfg)
    total_layers = unfreeze_last_qwen2vl_text_layers(
        model,
        num_last_layers=1,
        train_lm_head=False,
    )

    logger.info(f"Total text layers: {total_layers}")

    trainer.train()

    # Save the final model à la Hugging Face format (includes both model and processor, important for VLMs)
    final_save_path = output_dir / "fine_tuned_model"
    final_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_save_path)
    trainer.processing_class.save_pretrained(final_save_path)  # IMPORTANT for VLM
    logger.info(f"Saved final model checkpoint to {final_save_path}")

    # WARNING: This saves the entire model directory, which can be very large. Use with caution. (4-5 GB !!!!)
    # Log to wandb as artifact
    #artifact = wandb.Artifact(name="fine_tuned_model", type="model")
    #artifact.add_dir(str(final_save_path))  # Log the entire directory to preserve both model and processor
    #wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Fine-Tune Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name or path")
    parser.add_argument("--level", type=str, default="L1_single")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    #parser.add_argument("--target_layers", type=int, nargs='+', default=[26,27])
    
    args = parser.parse_args()

    # ──────────────────────────────────────────────
    # SEED
    # ──────────────────────────────────────────────
    set_global_seed(args.seed)

    # ──────────────────────────────────────────────
    # DEVICE
    # ──────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ──────────────────────────────────────────────
    # MODEL
    # ──────────────────────────────────────────────
    MODEL_ID = args.model_id

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map=device,
    )

    cfg = {
        "model_id":              MODEL_ID,
        "level":                 args.level,
        "num_train_epochs":      args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate":         args.learning_rate,
        "max_grad_norm":         args.max_grad_norm,
#        "target_layers":         args.target_layers,
        "d_model":               model.config.text_config.hidden_size,
        "device":                str(device),
        "seed":                  args.seed
    }



    # ──────────────────────────────────────────────
    # Processor & Collator
    # ──────────────────────────────────────────────

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    collator  = Spatial457Collator(processor)


    # ──────────────────────────────────────────────
    # Datasets
    # ──────────────────────────────────────────────

    train_dataset = DsAdapterSpatial457PerLevel(request_split=SPLIT_NAME_TRAIN, 
                                                request_level=args.level)
    eval_dataset  = DsAdapterSpatial457PerLevel(request_split=SPLIT_NAME_VALID,
                                                request_level=args.level)

    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        learning_rate=cfg["learning_rate"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_strategy="epoch",
        save_strategy="no",
        fp16=False,
        bf16=True,         # requires Ampere GPU (RTX 30xx, 40xx)
        logging_steps=1,
        report_to="wandb",  # ← Trainer logs loss/lr/eval metrics to wandb automatically
        remove_unused_columns=False,
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    main(args, cfg, model, trainer, collator)