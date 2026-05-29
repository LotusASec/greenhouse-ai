#!/usr/bin/env python3
"""
PlantVillage dataset download and organization.

Sources (tried in order):
  1. HuggingFace: plantvillage/plantdisease
  2. HuggingFace: huggan/plantvillage
  3. Manual Kaggle (prints instructions and exits 1)

Maps PlantVillage folder names to project class names using
models/disease/config/column_map.yaml (class_mapping section).

Usage:
  python scripts/download_dataset.py --output-dir data/
  python scripts/download_dataset.py --check
"""

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("download-dataset")

CONFIG_PATH = Path(__file__).parents[1] / "config" / "column_map.yaml"
SPLIT = (0.70, 0.15, 0.15)
MIN_TRAIN_IMAGES = 200
RANDOM_SEED = 42


def load_class_mapping(config_path: Path = CONFIG_PATH) -> dict[str, list[str]]:
    """Return {project_class: [plantvillage_folder, ...]}."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mapping: dict[str, list[str]] = {}
    for proj_cls, pv_cls in cfg["class_mapping"].items():
        mapping[proj_cls] = [pv_cls] if isinstance(pv_cls, str) else list(pv_cls)
    return mapping


def try_huggingface(output_dir: Path, class_mapping: dict) -> bool:
    """Attempt HuggingFace download. Returns True on success."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets library not installed — skipping HuggingFace")
        return False

    for repo_id in ("plantvillage/plantdisease", "huggan/plantvillage"):
        log.info("Trying HuggingFace: %s", repo_id)
        try:
            ds = load_dataset(repo_id, split="train", trust_remote_code=True)
            log.info("Downloaded %d images from %s", len(ds), repo_id)
            _organize_from_hf(ds, output_dir, class_mapping)
            return True
        except Exception as exc:
            log.warning("HuggingFace %s failed: %s", repo_id, exc)

    return False


def _organize_from_hf(ds, output_dir: Path, class_mapping: dict) -> None:
    """Sort HuggingFace dataset items into project class folders with train/val/test split."""
    from PIL import Image as PILImage

    # Build reverse mapping: pv_folder_name → project_class
    reverse: dict[str, str] = {}
    for proj_cls, pv_folders in class_mapping.items():
        for pv in pv_folders:
            reverse[pv.lower()] = proj_cls
            reverse[pv] = proj_cls

    # Collect images per project class
    buckets: dict[str, list] = {k: [] for k in class_mapping}

    # Detect label field
    label_field = None
    for candidate in ("label", "labels", "image_label", "category"):
        if candidate in ds.features:
            label_field = candidate
            break

    if label_field is None:
        log.error("Cannot find label field in dataset. Available: %s", list(ds.features.keys()))
        sys.exit(1)

    class_names_list = ds.features[label_field].names if hasattr(ds.features[label_field], "names") else []

    for item in ds:
        lbl = item[label_field]
        if isinstance(lbl, int) and class_names_list:
            lbl_name = class_names_list[lbl]
        else:
            lbl_name = str(lbl)

        proj_cls = reverse.get(lbl_name) or reverse.get(lbl_name.replace(" ", "_"))
        if proj_cls is None:
            continue
        buckets[proj_cls].append(item["image"])

    _write_splits(buckets, output_dir)


def try_local(local_dir: Path, output_dir: Path, class_mapping: dict) -> bool:
    """Organize from a local directory of PlantVillage images."""
    if not local_dir.exists():
        return False

    reverse: dict[str, str] = {}
    for proj_cls, pv_folders in class_mapping.items():
        for pv in pv_folders:
            reverse[pv.lower()] = proj_cls
            reverse[pv] = proj_cls

    buckets: dict[str, list[Path]] = {k: [] for k in class_mapping}

    for folder in local_dir.iterdir():
        if not folder.is_dir():
            continue
        proj_cls = reverse.get(folder.name) or reverse.get(folder.name.lower())
        if proj_cls is None:
            log.debug("Skipping unrecognized folder: %s", folder.name)
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + \
               list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
        buckets[proj_cls].extend(imgs)
        log.info("  %s → %s (%d images)", folder.name, proj_cls, len(imgs))

    _write_splits_from_paths(buckets, output_dir)
    return True


def _write_splits(buckets: dict, output_dir: Path) -> None:
    """Write PIL images from dict of lists to train/val/test dirs."""
    rng = random.Random(RANDOM_SEED)
    for proj_cls, images in buckets.items():
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * SPLIT[0])
        n_val   = int(n * SPLIT[1])
        splits = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }
        for split_name, split_imgs in splits.items():
            dest = output_dir / split_name / proj_cls
            dest.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(split_imgs):
                try:
                    if hasattr(img, "save"):
                        img.save(dest / f"{proj_cls}_{i:05d}.jpg")
                    else:
                        shutil.copy2(img, dest / f"{proj_cls}_{i:05d}.jpg")
                except Exception as exc:
                    log.debug("Skipping image %d: %s", i, exc)


def _write_splits_from_paths(buckets: dict, output_dir: Path) -> None:
    """Write from Path lists to train/val/test dirs."""
    rng = random.Random(RANDOM_SEED)
    for proj_cls, paths in buckets.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * SPLIT[0])
        n_val   = int(n * SPLIT[1])
        splits = {
            "train": paths[:n_train],
            "val":   paths[n_train:n_train + n_val],
            "test":  paths[n_train + n_val:],
        }
        for split_name, split_paths in splits.items():
            dest = output_dir / split_name / proj_cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                shutil.copy2(src, dest / src.name)


def print_counts(output_dir: Path) -> None:
    total = 0
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        log.info("--- %s ---", split.upper())
        for cls_dir in sorted(split_dir.iterdir()):
            n = len(list(cls_dir.glob("*")))
            log.info("  %-20s %5d images", cls_dir.name + ":", n)
            if split == "train":
                total += n
                if n < MIN_TRAIN_IMAGES:
                    log.warning("  ⚠ %s has %d < %d minimum", cls_dir.name, n, MIN_TRAIN_IMAGES)
    log.info("Total train images: %d", total)


def print_kaggle_instructions() -> None:
    print("""
Manual download required:
  pip install kaggle
  kaggle datasets download -d arjuntejaswi/plant-village
  unzip plant-village.zip -d models/disease/data/raw/
  python scripts/download_dataset.py --from-local models/disease/data/raw/
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and organize PlantVillage dataset")
    parser.add_argument("--output-dir",  default="data", help="Destination directory")
    parser.add_argument("--from-local",  default=None,   help="Skip download, organize from local folder")
    parser.add_argument("--check",       action="store_true", help="Just print existing counts")
    parser.add_argument("--config",      default=str(CONFIG_PATH))
    args = parser.parse_args()

    output_dir   = Path(args.output_dir)
    class_mapping = load_class_mapping(Path(args.config))
    log.info("Class mapping: %s", {k: v for k, v in class_mapping.items()})

    if args.check:
        print_counts(output_dir)
        return

    if args.from_local:
        log.info("Organizing from local directory: %s", args.from_local)
        ok = try_local(Path(args.from_local), output_dir, class_mapping)
        if not ok:
            log.error("Local directory not found: %s", args.from_local)
            sys.exit(1)
    else:
        ok = try_huggingface(output_dir, class_mapping)
        if not ok:
            log.error("All automatic downloads failed.")
            print_kaggle_instructions()
            sys.exit(1)

    log.info("Dataset organized in %s/", output_dir)
    print_counts(output_dir)


if __name__ == "__main__":
    main()
