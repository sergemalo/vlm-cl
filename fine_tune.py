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

def get_llm_layers(model):
    """
    Robustly retrieve the text decoder layers for Qwen2-VL.
    """
    try:
        return model.model.language_model.layers
    except AttributeError:
        raise RuntimeError(
            "Could not find text layers. Inspect model structure with print(model)"
        )

def unfreeze_qwen2vl(model,
                    train_merger: bool,
                    train_llm_top_n_layers: int, 
                    train_llm_head: bool):
    freeze_all_params(model)

    logger.info("Fine-tuning Strategy:")
    logger.info(f"Unfreeze vision-language merger: {train_merger}")
    logger.info(f"Unfreeze LLM Top n Layers:  {train_llm_top_n_layers}")
    logger.info(f"Unfreeze LLM Head:  {train_llm_head}")

    # Merger
    if train_merger:
        for param in model.model.visual.merger.parameters():
            param.requires_grad = True


    # LLM Top Layers
    # Qwen2-VL text decoder blocks live here in HF transformers
    if train_llm_top_n_layers > 0:
        llm_layers = get_llm_layers(model)
        total_llm_layers = len(llm_layers)

        if train_llm_top_n_layers > total_llm_layers:
            raise ValueError(
                f"Requested {train_llm_top_n_layers} LLM layers, but model only has {total_llm_layers}"
            )

        for layer in llm_layers[-train_llm_top_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True


    # Optional: unfreeze output head
    if train_llm_head:
        for p in model.lm_head.parameters():
            p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"After selection of model params to unfreeze:")
    logger.info(f"Nb Trainable params: {trainable:,}")
    logger.info(f"Total params:     {total:,}")
    logger.info(f"Percent:          {100 * trainable / total:.4f}%")


def main(args, cfg, model, trainer, collator):
    init_logging(args.log_level, output_dir)
    init_wandb(cfg)
    total_layers = unfreeze_qwen2vl(
        model,
        train_merger = True,
        train_llm_top_n_layers = 0, 
        train_llm_head = False)

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
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
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
    # This should already be true given your collator code, but verify:
    assert processor.tokenizer.pad_token_id == 151643


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
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine", # Default is linear
        warmup_steps = 10,
        max_grad_norm=cfg["max_grad_norm"],
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="no",
        fp16=False,
        bf16=True,         # requires Ampere GPU (RTX 30xx, 40xx)
        logging_strategy="steps",
        logging_steps=1,
        report_to="wandb",  # ← Trainer logs loss/lr/eval metrics to wandb automatically
        remove_unused_columns=False,
        weight_decay=0.01,
        optim="adamw_torch_fused",  # Default is adamw_torch_fused
        seed=cfg["seed"],
        data_seed=cfg["seed"]
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