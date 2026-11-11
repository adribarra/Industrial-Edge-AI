"""CPU backend: a simplified PatchCore with a ResNet18 (Roth et al. 2022).

How it works:
  1. A ResNet18 pretrained on ImageNet extracts features from two of its layers
     (layer2 and layer3).
  2. layer3 is enlarged to the size of layer2, both are smoothed with a 3x3 average and
     stacked together. That gives 384 numbers describing each small patch of the image.
  3. "Training" just means storing the patch descriptions of good images in a memory bank
     (a random 10% of them; the original method picks them more cleverly with a greedy
     coreset). There is no gradient training.
  4. For a new image, each patch is compared with the closest patch in the memory. These
     distances form the anomaly map, and the biggest one is the image's anomaly score
     (higher means more anomalous).

Input: float32 array (3, H, W), RGB, ImageNet-normalized (see app/preprocessing.py).
It runs on CPU only. The scores have no fixed scale, so the threshold must be chosen from
good validation images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from inference.base import BackendError, InferenceBackend, Prediction

WEIGHTS_FILE = "resnet18-f37072fd.pth"
BANK_FILE = "memory_bank.pt"


class CPUBackend(InferenceBackend):
    """Backend that runs the PatchCore-style model on the CPU."""

    name = "cpu"
    device = "cpu"

    def __init__(self, num_threads: Optional[int] = None):
        # num_threads limits how many CPU threads PyTorch uses (None = its default).
        self._net: Optional[torch.nn.Module] = None
        self._bank: Optional[torch.Tensor] = None
        self.input_size = None
        if num_threads:
            torch.set_num_threads(num_threads)

    @property
    def model_name(self) -> str:
        """Model name shown in the reports."""
        return "patchcore-resnet18"

    # ---- feature extraction -------------------------------------------------
    def _load_backbone(self, weights_path: Path) -> None:
        # Build a ResNet18 and load the pretrained weights from a file (no internet needed).
        if not weights_path.is_file():
            raise BackendError(f"Backbone weights not found: {weights_path}")
        net = torchvision.models.resnet18(weights=None)
        try:
            net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        except Exception as e:  # corrupt or mismatching file
            raise BackendError(f"Cannot load backbone weights: {e}") from e
        self._net = net.eval()

    @torch.no_grad()
    def _features(self, batch: torch.Tensor) -> torch.Tensor:
        """Turn a batch of images (N,3,H,W) into patch descriptions of shape (N, h, w, 384)."""
        n = self._net
        x = n.maxpool(n.relu(n.bn1(n.conv1(batch))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.avg_pool2d(l3, 3, 1, 1)
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([l2, l3], dim=1).permute(0, 2, 3, 1).contiguous()

    # ---- building the memory bank (normal images only) ---------------------
    def fit(self, weights_dir, images: Iterable[np.ndarray], keep_fraction: float = 0.1,
            seed: int = 0) -> int:
        """Build the memory bank from preprocessed GOOD images and return how many patches it holds.

        keep_fraction is the share of patches kept per image, picked randomly with the given seed.
        """
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        self._load_backbone(Path(weights_dir) / WEIGHTS_FILE)
        gen = torch.Generator().manual_seed(seed)
        chunks = []
        for img in images:
            f = self._features(torch.from_numpy(np.asarray(img, np.float32))[None])
            f = f.reshape(-1, f.shape[-1])
            k = max(1, int(len(f) * keep_fraction))
            chunks.append(f[torch.randperm(len(f), generator=gen)[:k]])
            self.input_size = tuple(img.shape[1:])
        if not chunks:
            raise BackendError("No images given to build the memory bank")
        self._bank = torch.cat(chunks)
        return len(self._bank)

    def save(self, model_dir) -> Path:
        """Save the memory bank in the model folder and return the file path."""
        if self._bank is None:
            raise BackendError("Nothing to save: memory bank is empty")
        out = Path(model_dir) / BANK_FILE
        torch.save({"bank": self._bank, "input_size": self.input_size}, out)
        return out

    # ---- InferenceBackend API ----------------------------------------------
    def load_model(self, model_path) -> None:
        """Load the weights and the memory bank from a model folder."""
        d = self._check_model_path(model_path)
        self._load_backbone(d / WEIGHTS_FILE)
        bank_path = d / BANK_FILE
        if not bank_path.is_file():
            raise BackendError(f"Memory bank not found: {bank_path} (run scripts/build_memory_bank.py)")
        try:
            data = torch.load(bank_path, map_location="cpu")
            self._bank = data["bank"].float()
            self.input_size = data.get("input_size")
        except Exception as e:
            raise BackendError(f"Cannot load memory bank: {e}") from e

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Prediction:
        """Score one preprocessed image (3,H,W float32) and return the score plus the anomaly map."""
        if self._net is None or self._bank is None:
            raise BackendError("Model not loaded; call load_model() first")
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[0] != 3 or arr.dtype != np.float32:
            raise BackendError(f"Expected float32 array (3,H,W), got {arr.dtype} {arr.shape}")
        try:
            f = self._features(torch.from_numpy(arr)[None])
            h, w = f.shape[1:3]
            patches = f.reshape(-1, f.shape[-1])
            d = torch.cdist(patches, self._bank).min(dim=1).values  # nearest normal patch
            amap = d.reshape(h, w)
        except Exception as e:
            raise BackendError(f"Inference failed: {e}") from e
        return Prediction(
            anomaly_score=float(amap.max()),
            anomaly_map=amap.numpy().astype(np.float32),
            metadata={"bank_size": int(len(self._bank)), "map_shape": (h, w)},
        )
