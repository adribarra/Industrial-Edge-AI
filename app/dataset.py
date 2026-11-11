"""Reads the MVTec AD dataset from disk.

The folders are expected to look like this:

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect_type>/*.png
    <root>/<category>/ground_truth/<defect_type>/<name>_mask.png

Every image gets a simple label: "good" means normal (0), any other folder name is a defect
(1). The pixel masks only exist for defective test images. I keep just their path, since they
are for locating the defect and not needed for the PASS/FAIL label.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_SPLITS = ("train", "test")


class DatasetError(Exception):
    """The dataset folder is missing or has no images."""


@dataclass(frozen=True)
class Sample:
    """One image of the dataset with its label and (if there is one) its mask."""

    path: Path
    split: str                        # "train" or "test"
    defect_type: str                  # "good" or the name of the defect
    label: int                        # 0 = normal, 1 = defective
    mask_path: Optional[Path] = None  # pixel-level ground truth, if there is one

    @property
    def label_name(self) -> str:
        """The label as text ("normal" or "anomaly"), used in the reports."""
        return "normal" if self.label == 0 else "anomaly"


def _is_image(p: Path) -> bool:
    # True for files with an image extension, False for anything else.
    return p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS


def _find_mask(category_dir: Path, defect_type: str, image: Path) -> Optional[Path]:
    # Look for the mask that belongs to a defective image. Good images have none.
    if defect_type == "good":
        return None
    mask = category_dir / "ground_truth" / defect_type / f"{image.stem}_mask.png"
    if mask.is_file():
        return mask
    logger.warning("Missing ground-truth mask for %s", image)
    return None


def load_category(dataset_dir, category: str, split: str = "test") -> List[Sample]:
    """List all the images of one category and split (train or test).

    Raises DatasetError if the folder is missing or empty. Files that are not images are
    skipped with a warning. Broken images are not detected here, because the images are not
    opened. That happens later in preprocessing.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    category_dir = Path(dataset_dir) / category
    split_dir = category_dir / split
    if not split_dir.is_dir():
        raise DatasetError(f"Dataset folder not found: {split_dir}")

    samples: List[Sample] = []
    for defect_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for img in sorted(defect_dir.iterdir()):
            if not _is_image(img):
                if img.is_file():
                    logger.warning("Skipping non-image file: %s", img)
                continue
            mask = _find_mask(category_dir, defect_type, img) if split == "test" else None
            samples.append(Sample(img, split, defect_type, label, mask))

    if not samples:
        raise DatasetError(f"No images found in {split_dir}")
    return samples
