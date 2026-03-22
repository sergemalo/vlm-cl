import argparse
import logging
import torch
import wandb
from datetime import datetime
from platform import processor
from sympy.stats import sample
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ds_adapter_spatial457 import *
from utils.general.seed_ctrl import set_global_seed
from utils.general.our_logging import init_logging

logger      = logging.getLogger(__name__)
date_prefix = datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
output_dir  = Path(f"output/{date_prefix}_eval")
output_dir.mkdir(parents=True, exist_ok=True)

class EvalResults:
    def __init__(self):
        self.resultsPerLevel = {}

    def add_result(self, level: str, success: bool):
        if level not in self.resultsPerLevel:
            self.resultsPerLevel[level] = {"total": 0, "success": 0}
        self.resultsPerLevel[level]["total"] += 1
        if success:
            self.resultsPerLevel[level]["success"] += 1

    def log_results(self):
        for level in sorted(self.resultsPerLevel.keys()):
            total = self.resultsPerLevel[level]["total"]
            success = self.resultsPerLevel[level]["success"]
            acc = success / total if total > 0 else 0.0
            logger.info(f"Level {level}: Accuracy = {acc:.2%} ({success}/{total})")

    def log_results_to_wandb(self):
        sorted_levels = sorted(self.resultsPerLevel.keys())
        sorted_accuracies = [self.resultsPerLevel[level]["success"] / self.resultsPerLevel[level]["total"] for level in sorted_levels]
        table = wandb.Table(data=[[l, a] for l, a in zip(sorted_levels, sorted_accuracies)],
                            columns=["level", "accuracy"])

        wandb.log({
            "accuracy_per_level": wandb.plot.bar(
                table,
                "level",      # x-axis
                "accuracy",   # y-axis
                title="Accuracy per Level"
            )
        })

def init_wandb(cfg: dict):        
    wandb.init(
        dir     = output_dir,
        project = "vlm-cl-qwen-2b",
        name    = date_prefix + "_eval",
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

def eval(cfg: dict):
    """
    Main evaluation function.
    """
    eval_results = EvalResults()

    # 1) Load dataset
    # (Assuming dataset loading and sampling code is here)
    eval_ds = DsAdapterSpatial457(request_split = SPLIT_NAME_TEST, max_level=cfg["max_level"])

    # 2) Load model and processor
    model_id = cfg["model_id"]
    logger.info(f"Loading model: {model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=cfg["device"],
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 3) WanB
    init_wandb(cfg)

    # 4) Evaluation loop
    for sample in tqdm(eval_ds, total=len(eval_ds)):

        image = sample['image_data']
        question = sample['question']
        target_answer = str(sample["answer"]).strip()
        level = sample['level']

        # TODO: Refactor to use different models, if needed
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a visual question answering assistant. Answer in one word."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gen_inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
        ).to(model.device)

        # Generate answer
        with torch.no_grad():
            output = model.generate(**gen_inputs, max_new_tokens=20)

        generated_ids = output[:, gen_inputs["input_ids"].shape[1]:]
        whole_pred_answer = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        filtered_pred_answer = whole_pred_answer[0].strip() if whole_pred_answer else ""  # Take only the first word as the answer

        # Evaluate answer
        success = eval_ds.eval_answer(filtered_pred_answer, target_answer)
        eval_results.add_result(level, success)

        if cfg["report_loss"]:
            # -------------------------
            # 2) Loss pass
            # -------------------------
            # Build full text = prompt + target answer
            full_text = prompt_text + target_answer

            loss_inputs = processor(
                text=[full_text],
                images=[image],
                return_tensors="pt",
            ).to(model.device)

            # Labels start as a copy of input_ids
            labels = loss_inputs["input_ids"].clone()

            # Mask prompt tokens so loss is only computed on answer tokens
            prompt_only_inputs = processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
            ).to(model.device)

            prompt_len = prompt_only_inputs["input_ids"].shape[1]
            labels[:, :prompt_len] = -100

            with torch.no_grad():
                loss_outputs = model(**loss_inputs, labels=labels)
                loss = loss_outputs.loss


        logger.debug(f"Qu: {question}")
        logger.debug(f"Ta: {target_answer.lower()}")
        #logger.debug(f"Whole generated answer: '{whole_pred_answer}'")
        logger.debug(f"Pa: {filtered_pred_answer.lower()}")
        logger.debug(f"S?: {success}")
        if cfg["report_loss"]:
            logger.debug(f"L : {loss}")

    # 5) Log results
    eval_results.log_results()
    eval_results.log_results_to_wandb()

    #wandb.save(log_file)
    wandb.finish()



def main():
    parser = argparse.ArgumentParser(description="VLM Evaluate Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name or path")
    #parser.add_argument("--dataset", type=str, default="scienceqa", help="Dataset name or path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_level", type=int, default=0, help="Max level of questions to evaluate (0 for all levels)")
    parser.add_argument("--report_loss", type=bool, default=False, help="Report loss in the debug log")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    args = parser.parse_args()

    init_logging(args.log_level, output_dir)

    set_global_seed(args.seed)

    # DEVICE SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = {
        "device": device,
        "model_id": args.model_id,
        "seed": args.seed,
        "max_level": args.max_level,
        "report_loss": args.report_loss,
        # "dataset_name": args.dataset,
    }

    eval(cfg)

if __name__ == "__main__":
    main()
