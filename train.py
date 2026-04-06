import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, AutoProcessor, Qwen2VLForConditionalGeneration

from utils.general.seed_ctrl import set_global_seed
from utils.general.our_logging import init_logging

from utils.data.dataset import DsAdapterSpatial457PerLevel, SPLIT_NAME_TRAIN, SPLIT_NAME_VALID
from utils.train.collator import Spatial457Collator
from utils.train.trainer import MyTrainer
from utils.eval.metrics import compute_metrics
from utils.cl.mlp_with_moe import MLPWithMoE
from utils.cl.adapter import Adapter
from utils.cl.expert_regularizer import ExpertRegularizer
 
import wandb
import logging
from datetime import datetime
from pathlib import Path
 
import argparse
 
 
logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_dir  = Path(f"output/{date_prefix}_train")
Path(output_dir).mkdir(parents=True, exist_ok=True)

past_level_mapping = {"L2_objects": "L1_single",
            "L3_2D_spatial": "L2_objects",
            "L4_pose": "L3_2D_spatial",
            "L5_6d_spatial": "L4_pose"}
 
 
def init_wandb(cfg: dict):
    wandb.init(
        dir     = output_dir,
        project = "vlm-cl-qwen-2b",
        entity = "vlm-cl",
        name    = date_prefix + "_train",
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
 
 
def set_trainable_param(model, cfg):
    for param in model.parameters():
        param.requires_grad = False
 
    model_dtype = next(p for p in model.parameters() if p.device.type != 'meta').dtype
    model_device = next(p for p in model.parameters() if p.device.type != 'meta').device
 
    # Load existing experts and routers if provided
    existing_experts_by_layer = {}
    existing_routers_by_layer = {}
    existing_alphas_by_layer  = {}
    if cfg.get("past_adapters_path") is not None:
        saved = torch.load(cfg["past_adapters_path"])
        for layer_idx in range(cfg["target_layers"][0], cfg["target_layers"][1]+1):
 
            # ── Experts ──────────────────────────────────────────
            prefix = f"model.language_model.layers.{layer_idx}.mlp.moe.experts."
            layer_state = {
                k[len(prefix):]: v
                for k, v in saved.items()
                if k.startswith(prefix)
            }
            experts = []
            expert_idx = 0
            while True:
                expert_keys = {
                    k[len(f"{expert_idx}."):]: v
                    for k, v in layer_state.items()
                    if k.startswith(f"{expert_idx}.")
                }
                if not expert_keys:
                    break
                expert = Adapter(d_model=cfg["d_model"], rank=cfg["moe_rank"])
                expert.load_state_dict(expert_keys)
                experts.append(expert)
                expert_idx += 1
            existing_experts_by_layer[layer_idx] = experts
 
            # ── Routers ──────────────────────────────────────────
            prefix = f"model.language_model.layers.{layer_idx}.mlp.moe.routers."
            layer_state = {
                k[len(prefix):]: v
                for k, v in saved.items()
                if k.startswith(prefix)
            }
            routers = []
            router_idx = 0
            while True:
                router_keys = {
                    k[len(f"{router_idx}."):]: v
                    for k, v in layer_state.items()
                    if k.startswith(f"{router_idx}.")
                }
                if not router_keys:
                    break
                # --- Infer dimensions from checkpoint ---
                weight = router_keys["weight"]
                out_features, in_features = weight.shape
 
                # --- Recreate router with correct shape ---
                router = nn.Linear(in_features, out_features)
                router.load_state_dict(router_keys)
                routers.append(router)
                router_idx += 1
            existing_routers_by_layer[layer_idx] = routers
 
            # ── Alphas ───────────────────────────────────────────
            prefix = f"model.language_model.layers.{layer_idx}.mlp.alphas."
            alphas = []
            alpha_idx = 0
            while True:
                key = f"{prefix}{alpha_idx}"
                if key not in saved:
                    break
                alpha = nn.Parameter(saved[key].clone())
                alphas.append(alpha)
                alpha_idx += 1
            existing_alphas_by_layer[layer_idx] = alphas
 
    for layer_idx in range(cfg["target_layers"][0], cfg["target_layers"][1]+1):
        original_mlp = model.model.language_model.layers[layer_idx].mlp
        existing_experts = existing_experts_by_layer.get(layer_idx, [])
        existing_routers = existing_routers_by_layer.get(layer_idx, [])
        existing_alphas  = existing_alphas_by_layer.get(layer_idx, None)
 
        # Parameter gradient set to True inside MoE class
        new_mlp = MLPWithMoE(
            mlp=original_mlp,
            d_model=cfg["d_model"],
            num_experts=cfg["num_experts"],
            rank=cfg["moe_rank"],
            top_k=cfg["top_k"],
            existing_experts=existing_experts,
            existing_routers=existing_routers,
            existing_alphas=existing_alphas,
        )
 
        new_mlp.moe.to(dtype=model_dtype, device=model_device)
        for alpha in new_mlp.alphas:
            alpha.data = alpha.data.to(dtype=model_dtype, device=model_device)
 
        model.model.language_model.layers[layer_idx].mlp = new_mlp
        model.model.language_model.layers[layer_idx].mlp.alphas[-1].requires_grad = True
 
 
def set_parameter_regularizer(model, cfg, collator):
    if cfg.get("past_adapters_path") is not None:
        # Retrieve dataset of previous level (args.level-1)
        prev_train_dataset = DsAdapterSpatial457PerLevel(request_split=SPLIT_NAME_TRAIN, 
                                                    request_level=past_level_mapping[args.level])
        old_task_dataloader = DataLoader(
            prev_train_dataset,
            batch_size=cfg["per_device_train_batch_size"],
            collate_fn=collator,
        )
        regularizer = ExpertRegularizer(
            model=model,
            old_task_dataloader=old_task_dataloader,  # dataloader from previous task
            lambda_reg=cfg["lambda_reg"],
            device=device,
        )
        trainer.set_regularizer(regularizer)
        
 
def main(args, cfg, model, trainer, collator):
    init_logging(args.log_level, output_dir)
    set_global_seed(args.seed)
    init_wandb(cfg)
    set_trainable_param(model, cfg)
    if args.with_regularization:
        set_parameter_regularizer(model, cfg, collator)
    
    trainer.train()
 
    # Save MoE parameters
    trainable_state_dict = {
        name: param
        for name, param in model.named_parameters()
        if "moe.experts" in name or "moe.routers" in name or "mlp.alphas" in name
    }
    save_path = Path(output_dir) / "moe_adapters.pt"
    torch.save(trainable_state_dict, save_path)
    logger.info(f"Saved MoE adapters to {save_path}")
 
    # Log to wandb as artifact
    artifact = wandb.Artifact(name="moe_adapters", type="model")
    artifact.add_file(str(save_path))
    wandb.log_artifact(artifact)
 
    wandb.finish()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Evaluate Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name or path")
    parser.add_argument("--level", type=str, default="L1_single")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
 
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--target_layers", type=int, nargs='+', default=[1,27])
    parser.add_argument("--past_adapters_path", type=str)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--moe_rank", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--lambda_reg", type=float, default=0.1) # Very strong parameter regularization
 
    # This flag makes with_regularization True if present
    parser.add_argument(
        "--with_regularization",
        action="store_true",  # sets True if flag is included
        help="Enable regularization"
    )
 
    # This allows you to explicitly disable it
    parser.add_argument(
        "--no_regularization",
        action="store_false", # sets False if flag is included
        dest="with_regularization", # stores in the same variable
        help="Disable regularization"
    )
 
    # Default value if no flag is given
    parser.set_defaults(with_regularization=True)
    
    args = parser.parse_args()
 
    # ──────────────────────────────────────────────
    # Config & Model
    # ──────────────────────────────────────────────
 
    MODEL_ID = args.model_id
 
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
 
    # DEVICE SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
 
    cfg = {
        "model_id":              MODEL_ID,
        "level":                 args.level,
        "epochs":                args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate":         args.learning_rate,
        "max_grad_norm":         args.max_grad_norm,
        "target_layers":         args.target_layers,
        "past_adapters_path":    args.past_adapters_path,
        "num_experts":           args.num_experts,
        "moe_rank":              args.moe_rank,
        "top_k":                 args.top_k,
        "with_regularization":   args.with_regularization,
        "lambda_reg":            args.lambda_reg,
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
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=1,
        learning_rate=cfg["learning_rate"],
        max_grad_norm=cfg["max_grad_norm"],
        eval_strategy="epoch",
        save_strategy="no",
        fp16=False,
        bf16=True,         # requires Ampere GPU (RTX 30xx, 40xx)
        logging_steps=250,
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