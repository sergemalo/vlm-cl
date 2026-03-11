import argparse
import logging
import torch
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


def eval(cfg: EvalConfig):
    pass
    # 1) Load dataset
    # (Assuming dataset loading and sampling code is here)

    # 2) Load model and processor
    #model_id = "Qwen/Qwen2-VL-2B-Instruct"
    #model = Qwen2VLForConditionalGeneration.from_pretrained(
    #    model_id,
    #    torch_dtype=torch.float16,



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
