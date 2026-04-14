import os
import hashlib
import logging
import json
from pathlib import Path
from typing import Dict, Iterable, Set
from PIL import Image
from huggingface_hub import snapshot_download


logger = logging.getLogger(__name__)

token = os.environ.get("HF_TOKEN")
if token is None:
    raise RuntimeError("Heille le comique, tu n'as pas setté ta variable d'environnement <HF_TOKEN>, c'est nécessaire pour downloader le DS de HF.")

DEFAULT_DS_REPO = "RyanWW/Spatial457"
SPLIT_NAME_TRAIN = "train"
SPLIT_NAME_VALID = "valid"
SPLIT_NAME_TEST = "test"

# TODO: DECIDE if we keep default split ratios


def get_images_questions_dirs(repo_dir: Path) -> tuple[Path, Path]:
    images_dir = repo_dir / "images"
    questions_dir = repo_dir / "questions"
    if images_dir.is_dir() and questions_dir.is_dir():
        return images_dir, questions_dir
    # sometimes nested
    for sub in repo_dir.iterdir():
        if sub.is_dir() and (sub / "images").is_dir() and (sub / "questions").is_dir():
            return sub / "images", sub / "questions"
    raise RuntimeError(f"Could not find images/ and questions/ under {repo_dir}")


def stable_split_from_image_name(
    image_name: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    salt: str = "spatial457_split_v1",
) -> str:
    """
    Deterministically assign an image filename to train / val / test.

    This is stable across:
    - script invocations
    - machines
    - OSes
    - Python versions
    """
    assert 0.0 < train_ratio < 1.0
    assert 0.0 <= val_ratio < 1.0
    assert train_ratio + val_ratio < 1.0

    key = f"{salt}:{image_name}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()

    # Stable bucket in [0, 1)
    bucket = int(digest[:16], 16) / float(16**16)

    if bucket < train_ratio:
        return SPLIT_NAME_TRAIN
    if bucket < train_ratio + val_ratio:
        return SPLIT_NAME_VALID
    return SPLIT_NAME_TEST


def get_images_names_set(
    images_dir: Path,
    request_split = SPLIT_NAME_TEST,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    salt: str = "spatial457_split_v1",
) -> Set[str]:
    """
    Scan the dataset and return the set of image filenames assigned to the split.
    """
    images_names: Set[str] = set()

    for image_name in images_dir.glob("*.png"):
        split = stable_split_from_image_name(
            image_name=image_name.stem,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            salt=salt,
        )
        if split == request_split:
            images_names.add(image_name.name)

    logger.info(f"Found {len(images_names)} in split {request_split}")
    return images_names


def load_images_into_memory(
    images_dir: Path,
    images_names: Iterable[str],
) -> Dict[str, Image.Image]:
    """
    Load images from disk into memory.

    Args
    ----
    images_dir:
        Directory containing the dataset images.

    images_names:
        Iterable of image filenames to load.

    Returns
    -------
    dict:
        {image_filename -> PIL.Image}
    """

    images_by_name: Dict[str, Image.Image] = {}

    logger.info(f"Loading {len(images_names)} images to Host RAM")
    for name in images_names:
        path = images_dir / name

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        images_by_name[name] = img

    return images_by_name


def normalize_answer(ans) -> str:
    ans = str(ans).strip().lower()

    # Remove simple trailing punctuation
    ans = ans.rstrip(" .,!?:;")

    bool_map = {
        "true": "yes",
        "false": "no",
        "yes": "yes",
        "no": "no",
    }
    if ans in bool_map:
        return bool_map[ans]

    return ans


def build_samples_from_questions_file(
    questions_file: Path,
    images_by_name: Dict[str, any]
) -> list[Dict[str, any]]:
    """
    Build samples from a single questions file.
    """

    logger.debug(f"Building samples from {questions_file}")
    with questions_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    samples: list[Dict[str, any]] = []
    level = questions_file.stem

    logger.debug(f"Processing {len(questions)} questions for level {level}")

    for q in questions:
        cur_image_name = q["image_filename"]

        # keep only questions whose image is already in the selected split
        if cur_image_name not in images_by_name:
            continue

        samples.append(
            {
                "image_data": images_by_name[cur_image_name],
                "level": level,
                "question": q["question"],
                "answer": normalize_answer(q["answer"]),
            }
        )

    logger.info(f"Built {len(samples)} samples from {questions_file.name}")
    return samples

def build_all_samples(
    questions_dir: Path,
    images_by_name: Dict[str, any],
    max_level: int = 0
) -> list[Dict[str, any]]:
    """
    Build samples from multiple question files and concatenate them.
    """
    all_samples: list[Dict[str, any]] = []
    valid_prefixes = [f"L{lvl}" for lvl in range(1, max_level + 1)] if max_level > 0 else None

    for questions_file in questions_dir.glob("*.json"):
        if valid_prefixes and not any(questions_file.stem.startswith(prefix) for prefix in valid_prefixes):
            continue
        if questions_file.stem in ["L4_occ", "L5_collision"]:
            logger.info(f"Skipping questions file {questions_file.name} as it corresponds to a level we want to exclude.")
            continue
        all_samples.extend(
            build_samples_from_questions_file(
                questions_file=questions_file,
                images_by_name=images_by_name,
            )
        )

    return all_samples

class DsAdapterSpatial457:
    def __init__(self, request_split=SPLIT_NAME_TEST, max_level: int = 0):

        logger.info("Download Spatial457 repo to HF cache...")
        repo_dir = Path(snapshot_download(DEFAULT_DS_REPO, repo_type="dataset"))
        images_dir, questions_dir = get_images_questions_dirs(repo_dir)
        logger.debug(f"Images dir: {images_dir}")


        images_names = get_images_names_set(images_dir, request_split)
        self.images_by_name = load_images_into_memory(images_dir, images_names)

        self.samples = build_all_samples(questions_dir, self.images_by_name, max_level)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def eval_answer(self, predicted_answer: str, target_answer: str) -> bool:
        """
        Evaluate the predicted answer against the target answer.

        For now, we do a simple string equality after stripping and lowercasing.
        This can be improved with more sophisticated metrics if needed.
        """

        # Handle boolean answers (e.g., True/False, Yes/No)
        if isinstance(predicted_answer, bool):
            predicted_answer = "yes" if predicted_answer else "no"
        if isinstance(target_answer, bool):
            target_answer = "yes" if target_answer else "no"

        if predicted_answer.lower() in ["true", "yes"]:
            predicted_answer = "yes"
        elif predicted_answer.lower() in ["false", "no"]:
            predicted_answer = "no"

        if target_answer.lower() in ["true", "yes"]:
            target_answer = "yes"
        elif target_answer.lower() in ["false", "no"]:
            target_answer = "no"
        
        return predicted_answer.strip().lower() == target_answer.strip().lower()
