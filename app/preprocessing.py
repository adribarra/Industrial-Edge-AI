"""Gets an image ready for the model. It does not depend on any specific model backend.

Steps: load the file, make it RGB, resize it, scale the pixels to 0-1, subtract the mean and
divide by the std, and put the channels first (3, height, width).

The defaults are the ImageNet mean/std and a 256x256 size. Both can be changed, because the
right values depend on the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessingError(Exception):
    """The image could not be loaded or has a wrong format."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for preprocessing: output size and the mean/std used to normalize."""

    width: int = 256
    height: int = 256
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self):
        # Reject obviously wrong settings right away.
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must have 3 values (RGB)")
        if any(s <= 0 for s in self.std):
            raise ValueError("std values must be positive")


def load_image(path) -> np.ndarray:
    """Read an image file and return it as an RGB uint8 array of shape (H, W, 3)."""
    p = Path(path)
    if not p.is_file():
        raise PreprocessingError(f"Image not found: {p}")
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)  # grayscale files also come out with 3 channels
    if bgr is None:
        raise PreprocessingError(f"Cannot decode image: {p}")
    # OpenCV gives BGR but the model expects RGB, so swap the channels
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def preprocess_array(rgb: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Turn an RGB uint8 image (H, W, 3) into a float32 array of shape (3, height, width)."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise PreprocessingError("Expected an array of shape (H, W, 3)")
    if rgb.dtype != np.uint8:
        raise PreprocessingError(f"Expected uint8 image, got {rgb.dtype}")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise PreprocessingError("Image has zero size")

    resized = cv2.resize(rgb, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray(cfg.mean, np.float32)) / np.asarray(cfg.std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1), dtype=np.float32)


def preprocess_image(path, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Load an image file and preprocess it. The result has shape (3, height, width)."""
    return preprocess_array(load_image(path), cfg)
