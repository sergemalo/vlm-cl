import argparse
import logging
from platform import processor
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ds_adapter_spatial457 import *
from seed_ctrl import set_global_seed

logger = logging.getLogger(__name__)


class EvalConfig:
    def __init__(
        self,
        device: torch.device,
        #model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        #dataset_name: str = "scienceqa",
        #images_dir: str = "./data/scienceqa/images",
        #output_file: str = None,
    ):
        self.device = device
        #self.model_name = model_name
        #self.dataset_name = dataset_name
        #self.images_dir = images_dir
        #self.output_file = output_file
        #self.max_new_tokens = max_new_tokens


class EvalResults:
    def __init__(self):
        self.samplesPerLevel = {}
        self.successPerLevel = {}

    def add_result(self, level: int, success: bool):
        if level not in self.samplesPerLevel:
            self.samplesPerLevel[level] = 0
            self.successPerLevel[level] = 0
        self.samplesPerLevel[level] += 1
        if success:
            self.successPerLevel[level] += 1

    def log_results(self):
        for level in sorted(self.samplesPerLevel.keys()):
            total = self.samplesPerLevel[level]
            success = self.successPerLevel[level]
            acc = success / total if total > 0 else 0.0
            logger.info(f"Level {level}: Accuracy = {acc:.2%} ({success}/{total})")

def eval(cfg: EvalConfig):
    eval_results = EvalResults()

    # 1) Load dataset
    # (Assuming dataset loading and sampling code is here)
    eval_ds = DsAdapterSpatial457(request_split = SPLIT_NAME_TEST)

    # 2) Load model and processor
    # TODO: Move model loading to model factory funciton
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    logger.info(f"Loading model: {model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=cfg.device,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 3) Evaluation loop
    for sample in eval_ds:

        # (Assuming sample contains 'image', 'question', 'answer', and 'level')
        image = sample['image']
        question = sample['question']
        target_answer = sample['answer']
        level = sample['level']

        # (Assuming model inference code is here to get 'predicted_answer')
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=100)
        predicted_answer = processor.decode(output[0], skip_special_tokens=True)
 
        success = (predicted_answer == target_answer)
        logger.info(f"Predicted Answer: {predicted_answer}; Target Answer: {target_answer}; Success: {success}")
        eval_results.add_result(level, success)

    # 4) Log results
    eval_results.log_results()




def main():
    parser = argparse.ArgumentParser(description="VLM Evaluate Script")
    #parser.add_argument("--model", type=str, required=True, help="Model name or path")
    #parser.add_argument("--dataset", type=str, default="scienceqa", help="Dataset name or path")
    #parser.add_argument("--images_dir", type=str, default="./data/scienceqa/images", help="Directory containing images")
    #parser.add_argument("--output_file", type=str, default=None, help="File to save detailed results (JSONL)")
    #parser.add_argument("--max_new_tokens", type=int, default=16, help="Max new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    args = parser.parse_args()

    # LOGGING SETUP
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # SEED SETUP
    set_global_seed(args.seed)

    # DEVICE SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    cfg = EvalConfig(
        device=device,
        #model_name="Qwen/Qwen2-VL-2B-Instruct",
        #dataset_name="scienceqa",
        #images_dir="./data/scienceqa/images",
        #output_file=args.output_file,
        #max_new_tokens=16,
       # log_level=args.log_level,
    )

    eval(cfg)

if __name__ == "__main__":
    main()
