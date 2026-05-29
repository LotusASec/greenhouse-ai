#!/usr/bin/env python3
"""
Image simulator for Greenhouse AI — Phase 2B.

Samples real PlantVillage images according to node's
class_distribution config and returns base64-encoded images.
Class names always read from disease column_map.yaml.

Usage:
  python image_simulator.py --config config/node1.yaml
                            --dataset ../models/disease/data/
"""

import argparse
import base64
import logging
import os
import random
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("image-simulator")

DISEASE_CONFIG_PATH = Path(__file__).parent.parent / "models" / "disease" / "config" / "column_map.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ImageSimulator:
    """Samples PlantVillage images by class distribution from node config."""

    def __init__(self, config_path: str, dataset_path: str,
                 disease_config_path: Path = DISEASE_CONFIG_PATH) -> None:
        node_cfg = _load_yaml(Path(config_path))
        self.node_id: str = node_cfg["node_id"]
        self._distribution: dict[str, float] = node_cfg.get("image_class_distribution", {})

        # Validate distribution
        total = sum(self._distribution.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"image_class_distribution sums to {total:.3f}, expected ≈1.0")

        # Class names from disease column_map.yaml
        disease_cfg = _load_yaml(disease_config_path)
        self._valid_classes: list[str] = list(disease_cfg["class_mapping"].keys())

        # Validate distribution keys against valid class names
        invalid = set(self._distribution) - set(self._valid_classes)
        if invalid:
            raise ValueError(f"Unknown classes in distribution: {invalid}. Valid: {self._valid_classes}")

        # Index images per class from the dataset path
        self._image_index: dict[str, list[Path]] = {}
        ds = Path(dataset_path)
        # Check all splits for images
        for split in ("train", "val", "test"):
            for cls in self._valid_classes:
                cls_dir = ds / split / cls
                if cls_dir.exists():
                    imgs = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")) + \
                           sorted(cls_dir.glob("*.JPG")) + sorted(cls_dir.glob("*.PNG"))
                    if cls not in self._image_index:
                        self._image_index[cls] = []
                    self._image_index[cls].extend(imgs)

        if not any(self._image_index.values()):
            raise FileNotFoundError(
                f"No images found in {dataset_path}. "
                "Run: python scripts/download_dataset.py first."
            )

        for cls, imgs in self._image_index.items():
            log.info("  %s: %d images indexed", cls, len(imgs))

        self._rng = random.Random(42)
        log.info("ImageSimulator ready — node: %s, classes: %s",
                 self.node_id, self._valid_classes)

    def _sample_class(self) -> str:
        """Sample a class name according to the configured distribution."""
        classes = list(self._distribution.keys())
        weights = [self._distribution[c] for c in classes]
        return self._rng.choices(classes, weights=weights, k=1)[0]

    def sample(self, n: int = 1) -> list[dict]:
        """
        Return n samples, each a dict:
          { image_b64, ground_truth_class, node_id, timestamp }
        """
        results = []
        for _ in range(n):
            cls = self._sample_class()
            imgs = self._image_index.get(cls, [])
            if not imgs:
                log.warning("No images for class %s — skipping", cls)
                continue
            img_path = self._rng.choice(imgs)
            try:
                image_b64 = self._encode_image(img_path)
            except Exception as exc:
                log.warning("Failed to encode %s: %s", img_path, exc)
                continue
            results.append({
                "image_b64":          image_b64,
                "ground_truth_class": cls,
                "node_id":            self.node_id,
                "timestamp":          datetime.now(timezone.utc).isoformat(),
            })
        return results

    def _encode_image(self, path: Path) -> str:
        """Load image, resize to 224×224, return base64 string."""
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB").resize((224, 224))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            # Pillow not installed — return raw file bytes
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

    def run(self, interval_seconds: float = 1.0) -> None:
        """Continuous loop: sample + log to stdout. Ctrl-C to stop."""
        log.info("Starting image emit loop — interval: %.1fs", interval_seconds)
        while True:
            try:
                for item in self.sample(1):
                    log.info(
                        "%s | %s | class=%s | img_len=%d bytes",
                        item["node_id"], item["timestamp"],
                        item["ground_truth_class"], len(item["image_b64"]),
                    )
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                log.info("ImageSimulator stopped")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Greenhouse image simulator")
    parser.add_argument("--config",  default=os.getenv("NODE_PROFILE_PATH", "config/node1.yaml"))
    parser.add_argument("--dataset", default=os.getenv("DISEASE_DATA_PATH", "../models/disease/data"))
    parser.add_argument("--count",   type=int, default=0, help="Sample N images and exit (0=continuous)")
    parser.add_argument("--interval",type=float, default=1.0)
    args = parser.parse_args()

    sim = ImageSimulator(config_path=args.config, dataset_path=args.dataset)

    if args.count > 0:
        samples = sim.sample(args.count)
        log.info("Sampled %d images", len(samples))
        from collections import Counter
        dist = Counter(s["ground_truth_class"] for s in samples)
        for cls, cnt in sorted(dist.items()):
            log.info("  %-20s %4d  (%.1f%%)", cls, cnt, cnt / len(samples) * 100)
    else:
        sim.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
