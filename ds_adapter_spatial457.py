from huggingface_hub import snapshot_download
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, Set
from PIL import Image


logger = logging.getLogger(__name__)

token = os.environ.get("HF_TOKEN")
if token is None:
    raise RuntimeError("Heille le comique, tu n'as pas setté ta variable d'environnement <HF_TOKEN>, c'est nécessaire pour downloader le DS de HF.")


DEFAULT_DS_REPO = "RyanWW/Spatial457"
SPLIT_NAME_TRAIN = "train"
SPLIT_NAME_VALID = "valid"
SPLIT_NAME_TEST = "test"


def pick_images_questions_dirs(repo_dir: Path) -> tuple[Path, Path]:
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
            images_names.add(image_name)

    logger.info(f"Found {len(images_names)} in split {request_split}")
    return images_names

from pathlib import Path


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



def build_samples_from_questions_file(
    questions_file: Path,
    images_by_name: Dict[str, any]
) -> List[Dict[str, any]]:
    """
    Build samples from one question file.

    Returns a list of dict with:
      - image_data
      - level
      - question
      - answer
    """
    rows = _load_json_or_jsonl(questions_file)

    if default_level_name is None:
        default_level_name = question_file.stem

    samples: List[Dict[str, Any]] = []

    for row in rows:
        image_name = _extract_image_name(row)

        if image_name not in images_by_name:
            continue

        samples.append(
            {
                "image_data": images_by_name[image_name],
                "level": _extract_level_name(row, default_level_name),
                "question": _extract_question(row),
                "answer": _extract_answer(row),
            }
        )

    return samples


def build_all_samples(
    questions_dir: Path,
    images_by_name: Dict[str, any],
) -> list[Dict[str, any]]:
    """
    Build samples from multiple question files and concatenate them.
    """
    all_samples: list[Dict[str, any]] = []

    for questions_file in questions_dir.glob("*.json"):
        all_samples.extend(
            build_samples_from_questions_file(
                question_files=questions_file,
                images_by_name=images_by_name,
            )
        )

    return all_samples

class DsAdapterSpatial457:
    def __init__(self, request_split=SPLIT_NAME_TEST):

        logger.info("Download Spatial457 repo to HF cache...")
        repo_dir = Path(snapshot_download(DEFAULT_DS_REPO, repo_type="dataset"))
        images_dir, questions_dir = pick_images_questions_dirs(repo_dir)
        logger.debug(f"Images dir: {images_dir}")


        images_names = get_images_names_set(images_dir, request_split)
        self.images_by_name = load_images_into_memory(images_dir, images_names)

        self.samples = build_all_samples(questions_dir, self.images_by_name)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pass
        #ex = self.images[idx]

        #image = ex["image"]
        #if isinstance(image, str) and self.load_images:
        #    image = Image.open(image).convert("RGB")

#        return {
#            "image": image,
#            "question": ex["question"],
#            "answer": ex.get("answer", ex.get("gt")),
#            "image_id": ex["image"],
#            "level": ex["level"],
#           }