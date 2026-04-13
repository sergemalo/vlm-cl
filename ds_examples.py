"""
find_common_images.py

Finds images that appear in ALL 7 difficulty configs of Spatial457,
saves up to 10 such images as img_<id>.png, and writes their questions
to questions_<id>.txt.

Requirements:
    pip install huggingface_hub pillow
"""

import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ID    = "RyanWW/Spatial457"
REPO_TYPE  = "dataset"
MAX_IMAGES = 10
OUTPUT_DIR = Path(".")

QUESTION_FILES = {
    "L1_single"    : "questions/L1_single.json",
    "L2_objects"   : "questions/L2_objects.json",
    "L3_2d_spatial": "questions/L3_2D_spatial.json",
    "L4_occ"       : "questions/L4_occ.json",
    "L4_pose"      : "questions/L4_pose.json",
    "L5_6d_spatial": "questions/L5_6d_spatial.json",
    "L5_collision" : "questions/L5_collision.json",
}

# ── Helper: extract a flat list of record dicts from any JSON layout ──────────

def extract_records(data, config: str) -> list[dict]:
    """
    Handles:
      A. List of dicts       [ {"image_filename": ..., "question": ...}, ... ]
      B. COCO-style dict     { "info": {...}, "questions": [...], ... }
      C. Dict of dicts       { "id_001": {"image_filename": ..., ...}, ... }
      D. Dict of lists       { "image_filename": [...], "question": [...], ... }
    """

    # ── Case A: root is a list ─────────────────────────────────────────────────
    if isinstance(data, list):
        if not data:
            return []
        if not isinstance(data[0], dict):
            raise TypeError(f"[{config}] List elements are {type(data[0]).__name__}, expected dict.")
        print(f"    layout A – list of dicts  |  keys: {list(data[0].keys())}")
        return data

    if not isinstance(data, dict):
        raise TypeError(f"[{config}] JSON root is {type(data).__name__}, expected list or dict.")

    # ── Case B: COCO-style – look for a list-valued key whose items are dicts
    #    with "image_filename" or "question" ────────────────────────────────────
    RECORD_KEYS = {"image_filename", "question", "answer", "question_index"}
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if RECORD_KEYS & set(value[0].keys()):   # at least one field matches
                print(f"    layout B – COCO-style dict  |  records key: '{key}'  |  "
                      f"record keys: {list(value[0].keys())}")
                return value

    # ── Case C: dict-of-dicts (every value is a dict with record fields) ───────
    first_val = next(iter(data.values()))
    if isinstance(first_val, dict) and (RECORD_KEYS & set(first_val.keys())):
        print(f"    layout C – dict of dicts  |  inner keys: {list(first_val.keys())}")
        return list(data.values())

    # ── Case D: columnar dict-of-lists ─────────────────────────────────────────
    if isinstance(first_val, list):
        print(f"    layout D – columnar dict  |  columns: {list(data.keys())}")
        keys   = list(data.keys())
        length = len(first_val)
        return [{k: data[k][i] for k in keys} for i in range(length)]

    # ── Unknown – dump structure so the user can report it ────────────────────
    print(f"\n[{config}] UNKNOWN structure. Top-level keys and value types:")
    for k, v in data.items():
        desc = f"list[{type(v[0]).__name__}] len={len(v)}" if isinstance(v, list) else \
               f"dict keys={list(v.keys())[:6]}"           if isinstance(v, dict)  else \
               repr(v)[:80]
        print(f"  '{k}': {desc}")
    raise TypeError(f"[{config}] Could not determine record layout. See dump above.")


# ── Step 1: Load all question JSONs ───────────────────────────────────────────

image_index: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))

print("Downloading question JSONs from HuggingFace Hub...\n")

for config, hf_path in QUESTION_FILES.items():
    print(f"  {config}")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        repo_type=REPO_TYPE,
    )
    with open(local_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = extract_records(raw, config)

    for item in records:
        fname = item["image_filename"]
        q     = item["question"]
        a     = str(item["answer"])
        image_index[fname][config].append((q, a))

    unique_imgs = len({item["image_filename"] for item in records})
    print(f"    {len(records):5d} questions  |  {unique_imgs:4d} unique images\n")

# ── Step 2: Find images present in ALL 7 configs ──────────────────────────────

all_configs   = set(QUESTION_FILES.keys())
common_images = sorted(
    fname
    for fname, config_map in image_index.items()
    if set(config_map.keys()) == all_configs
)
print(f"Images in all {len(all_configs)} configs: {len(common_images)}")

if not common_images:
    print("Falling back to images present in >=5 configs...")
    common_images = sorted(
        fname
        for fname, config_map in image_index.items()
        if len(config_map) >= 5
    )
    print(f"Images in >=5 configs: {len(common_images)}")

selected = common_images[:MAX_IMAGES]
print(f"Selecting {len(selected)} image(s).\n")

# ── Step 3: Download images and write outputs ─────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for idx, fname in enumerate(selected):
    local_img_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"images/{fname}",
        repo_type=REPO_TYPE,
    )

    out_img = OUTPUT_DIR / f"img_{idx}.png"
    Image.open(local_img_path).convert("RGB").save(out_img)

    out_txt    = OUTPUT_DIR / f"questions_{idx}.txt"
    config_map = image_index[fname]
    total_q    = sum(len(v) for v in config_map.values())

    lines = [
        f"Image   : {fname}",
        f"Index   : {idx}",
        f"Configs : {sorted(config_map.keys())}",
        f"Total Q : {total_q}",
        "=" * 60,
    ]
    for config in QUESTION_FILES:
        if config not in config_map:
            continue
        lines.append(f"\n--- {config} ---")
        for q_idx, (question, answer) in enumerate(config_map[config], start=1):
            lines.append(f"  Q{q_idx}: {question}")
            lines.append(f"  A{q_idx}: {answer}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[{idx}] {fname}")
    print(f"      image     -> {out_img}")
    print(f"      questions ({total_q}) -> {out_txt}\n")

print("Done.")