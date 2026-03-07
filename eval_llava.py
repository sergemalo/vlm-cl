#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

# Default dataset repo to download (scriptless)
DEFAULT_DS_REPO = "RyanWW/Spatial457"
# Model you chose
DEFAULT_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"


def normalize_answer(s):
    if isinstance(s, bool):
        return "yes" if s else "no"

    if s is None:
        return ""

    s = str(s)
    s = s.strip().lower()

    # map common boolean words
    if s in ["true", "yes"]:
        return "yes"
    if s in ["false", "no"]:
        return "no"

    # remove articles
    import re
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_images_questions_dirs(repo_dir: Path) -> Tuple[Path, Path]:
    images_dir = repo_dir / "images"
    questions_dir = repo_dir / "questions"
    if images_dir.is_dir() and questions_dir.is_dir():
        return images_dir, questions_dir
    # sometimes nested
    for sub in repo_dir.iterdir():
        if sub.is_dir() and (sub / "images").is_dir() and (sub / "questions").is_dir():
            return sub / "images", sub / "questions"
    raise RuntimeError(f"Could not find images/ and questions/ under {repo_dir}")


def level_json_files(questions_dir: Path) -> Dict[int, Path]:
    out = {}
    for p in questions_dir.glob("L*.json"):
        # filenames like L1_single.json, L2_*.json, etc.
        m = re.match(r"^L(\d+)_.*\.json$", p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    return out


def load_questions(json_path: Path) -> List[dict]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data and isinstance(data["questions"], list):
        return data["questions"]
    raise RuntimeError(f"Unexpected JSON structure in {json_path} (top keys: {list(data) if isinstance(data, dict) else type(data)})")


def resolve_image_path(images_dir: Path, image_filename: str) -> Path:
    p = images_dir / image_filename
    if p.exists():
        return p
    # fallback: search by basename
    matches = list(images_dir.rglob(Path(image_filename).name))
    if matches:
        return matches[0]
    return p  # will fail upstream with clear message


@torch.inference_mode()
def generate_answer(
    model,
    processor,
    image: Image.Image,
    question: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    # LLaVA-Next uses chat-style formatting. This is the most robust pattern:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out[0][input_len:]
    pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--max_level", type=int, required=True)
    ap.add_argument("-s", "--samples", type=int, default=200, help="samples per level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ds_repo", type=str, default=DEFAULT_DS_REPO)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--use_4bit", action="store_true", help="recommended for 16GB")
    ap.add_argument("--save_jsonl", type=str, default="", help="optional path to save preds")
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) Download Spatial457 repo to HF cache
    repo_dir = Path(snapshot_download(repo_id=args.ds_repo, repo_type="dataset"))
    images_dir, questions_dir = pick_images_questions_dirs(repo_dir)
    lvl_map = level_json_files(questions_dir)

    print(f"[OK] Dataset repo: {args.ds_repo}")
    print(f"[OK] Local snapshot: {repo_dir}")
    print(f"[OK] Images dir: {images_dir}")
    print(f"[OK] Questions dir: {questions_dir}")

    # 2) Load questions per level + sample
    per_level = {}
    for lvl in range(1, args.max_level + 1):
        if lvl not in lvl_map:
            print(f"[WARN] Missing json for L{lvl}")
            continue
        qs = load_questions(lvl_map[lvl])
        per_level[lvl] = qs
        print(f"[INFO] L{lvl}: {len(qs)} questions ({lvl_map[lvl].name})")

    sampled = {}
    for lvl, qs in per_level.items():
        k = min(args.samples, len(qs))
        sampled[lvl] = random.sample(qs, k)

    # 3) Load model (4-bit recommended on 16GB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quant_cfg = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    print(f"[INFO] Loading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype if not args.use_4bit else None,
        quantization_config=quant_cfg,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    if not args.use_4bit:
        model.to(device)
    model.eval()

    # 4) Eval loop (exact match after normalization)
    total = 0
    correct = 0
    per_level_acc = {}

    out_f = open(args.save_jsonl, "w", encoding="utf-8") if args.save_jsonl else None

    for lvl in sorted(sampled.keys()):
        lvl_total = 0
        lvl_correct = 0
        print(f"\n=== Evaluating L{lvl} on {len(sampled[lvl])} samples ===")

        for ex in sampled[lvl]:
            q = ex.get("question", "")
            gt = ex.get("answer", "")
            img_name = ex.get("image_filename", "")

            if not q or not img_name:
                continue

            img_path = resolve_image_path(images_dir, img_name)
            if not img_path.exists():
                print(f"[WARN] Image not found: {img_name}")
                continue

            image = Image.open(img_path).convert("RGB")
            pred = generate_answer(
                model=model,
                processor=processor,
                image=image,
                question="Answer in one word." + q,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            ok = normalize_answer(pred) == normalize_answer(gt)

            print(f"Example: pred='{pred}', gt='{gt}', correct={ok}")

            total += 1
            lvl_total += 1
            correct += int(ok)
            lvl_correct += int(ok)

            if out_f:
                out_f.write(json.dumps({
                    "level": lvl,
                    "image": img_name,
                    "question": q,
                    "gt": gt,
                    "pred": pred,
                    "correct": bool(ok),
                }, ensure_ascii=False) + "\n")

        acc = (lvl_correct / lvl_total) if lvl_total else 0.0
        per_level_acc[lvl] = acc
        print(f"[RESULT] L{lvl} acc: {acc:.3f} ({lvl_correct}/{lvl_total})")

    if out_f:
        out_f.close()

    overall = (correct / total) if total else 0.0
    print("\n=== Summary ===")
    for lvl in sorted(per_level_acc.keys()):
        print(f"L{lvl}: {per_level_acc[lvl]:.3f}")
    print(f"Overall: {overall:.3f} ({correct}/{total})")


if __name__ == "__main__":
    main()