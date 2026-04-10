import torch
import torch.nn as nn
from transformers import TrainingArguments, AutoProcessor, Qwen2VLForConditionalGeneration
 
from seed_ctrl import set_global_seed
from eval import init_logging
 
from utils.data.dataset import DsAdapterSpatial457PerLevel, SPLIT_NAME_VALID
from utils.train.collator import Spatial457Collator
from utils.train.trainer import MyTrainer
from utils.train.trainer_w_classifier import MyTrainerWithClassifier
from utils.eval.metrics import compute_metrics
from utils.cl.mlp_with_moe import MLPWithMoE
from utils.cl.adapter import Adapter
 
import wandb
import logging
from datetime import datetime
from pathlib import Path
import re
 
import argparse
 
 
logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M")
output_dir  = f"output/{date_prefix}"
Path(output_dir).mkdir(parents=True, exist_ok=True)
 
 
def init_wandb(cfg: dict):
    wandb.init(
        dir     = output_dir,
        project = "vlm-cl-qwen-2b",
        entity  = "vlm-cl",
        name    = date_prefix + "_eval",
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
 
 
def infer_num_experts(state_dict, layer_idx=1):
    prefix = f"model.language_model.layers.{layer_idx}.mlp.moe.experts."
 
    expert_indices = set()
 
    for k in state_dict.keys():
        if k.startswith(prefix):
            # Extract the expert index i
            # pattern: experts.i.
            match = re.search(r"experts\.(\d+)\.", k)
            if match:
                expert_indices.add(int(match.group(1)))
 
    if not expert_indices:
        raise ValueError(f"No experts found for layer {layer_idx}")
 
    return max(expert_indices) + 1
 
 
def infer_rank(state_dict, layer_idx=1):
    prefix = f"model.language_model.layers.{layer_idx}.mlp.moe.experts."
 
    for k, v in state_dict.items():
        if k.startswith(prefix) and k.endswith("down_proj.weight"):
            rank = v.shape[0]   # [rank, d_model]
            return rank
 
    raise ValueError(f"Could not find down_proj.weight for layer {layer_idx}")
 
 
def extract_level_id(name: str) -> int:
    return int(name.split("_")[0][1:])-1
 
 
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
            alphas = []
            alpha_idx = 0
            while True:
                key = f"model.language_model.layers.{layer_idx}.mlp.alphas.{alpha_idx}"
                if key not in saved:
                    break
                alpha = nn.Parameter(saved[key].clone())
                alphas.append(alpha)
                alpha_idx += 1
            existing_alphas_by_layer[layer_idx] = alphas
 
    mlps_with_moe = []
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
            mode="eval",
            level_id=extract_level_id(cfg["level"]),
        )
 
        new_mlp.moe.to(dtype=model_dtype, device=model_device)
        for alpha in new_mlp.alphas:
            alpha.data = alpha.data.to(dtype=model_dtype, device=model_device)
 
        model.model.language_model.layers[layer_idx].mlp = new_mlp
        for alpha in model.model.language_model.layers[layer_idx].mlp.alphas:
            alpha.requires_grad = False

        mlps_with_moe.append(new_mlp)
        

    #logger.info(f"Set {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters to require grad")
    #logger.info(f"Existing experts by layer: { {layer: len(experts) for layer, experts in existing_experts_by_layer.items()} }")
    return mlps_with_moe
 
        
 
def main(args, cfg, model, trainer):
    init_logging(args.log_level)
    set_global_seed(args.seed)
    init_wandb(cfg)
    mlps_with_moe = set_trainable_param(model, cfg)

    if type(trainer) is MyTrainerWithClassifier:
        trainer.set_mlps_with_moe(mlps_with_moe)
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    logger.info("Starting evaluation...")
    trainer.evaluate()
 
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
 
    parser.add_argument("--target_layers", type=int, nargs='+', default=[1,27])
    parser.add_argument("--past_adapters_path", type=str)
    parser.add_argument("--classifier_path", type=str, default="")
    parser.add_argument("--top_k", type=int, default=2)
    
    args = parser.parse_args()
 
    # ──────────────────────────────────────────────
    # Config & Model
    # ──────────────────────────────────────────────
 
    MODEL_ID = args.model_id
 
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
 
    # DEVICE SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
 
    state_dict = torch.load(args.past_adapters_path)
    num_experts = infer_num_experts(state_dict)
    moe_rank = infer_rank(state_dict)
 
    cfg = {
        "model_id":              MODEL_ID,
        "level":                 args.level,
        "target_layers":         args.target_layers,
        "past_adapters_path":    args.past_adapters_path,
        "num_experts":           num_experts,
        "moe_rank":              moe_rank,
        "top_k":                 args.top_k,
        "d_model":               model.config.text_config.hidden_size,
        "device":                str(device),
        "seed":                  args.seed
    }
 
    # ──────────────────────────────────────────────
    # Processor
    # ──────────────────────────────────────────────
 
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    collator  = Spatial457Collator(processor)
 
 
    # ──────────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────────
 
    eval_dataset  = DsAdapterSpatial457PerLevel(request_split=SPLIT_NAME_VALID,
                                                request_level=args.level)
 
    # ──────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────
 
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=False,
        bf16=True,         # requires Ampere GPU (RTX 30xx, 40xx)
        logging_steps=250,
        report_to="wandb",  # ← Trainer logs loss/lr/eval metrics to wandb automatically
        remove_unused_columns=False,
    )
    
    if args.classifier_path:
        trainer = MyTrainerWithClassifier(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.load_classifier(args.classifier_path)
    else:
        trainer = MyTrainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
 
    main(args, cfg, model, trainer)