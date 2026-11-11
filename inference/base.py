"""The common interface between the app and any inference backend.

The app (inspection, benchmark, reports) only knows about this file. The code that is specific
to a library (PyTorch, OpenVINO, the FPGA AI Suite) goes inside a concrete backend.

predict() takes a preprocessed float32 array of shape (3, H, W), see app/preprocessing.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BackendError(Exception):
    """Loading the model or running inference failed inside a backend."""


@dataclass
class Prediction:
    """What a backend returns for one image.

    anomaly_score: one number for the whole image, higher means more anomalous. The scale
        depends on the model, so the threshold has to be chosen for each model.
    prediction: "normal" or "anomaly" if the backend decides by itself, otherwise None.
    anomaly_map: optional (H, W) map showing where the anomaly is. Finding the defect
        (pixel level) is a different task from deciding if the image is bad (image level).
    metadata: any extra information from the backend.
    """

    anomaly_score: float
    prediction: Optional[str] = None
    anomaly_map: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Make sure the values make sense as soon as the object is created.
        self.anomaly_score = float(self.anomaly_score)
        if not np.isfinite(self.anomaly_score):
            raise ValueError("anomaly_score must be finite")
        if self.prediction not in (None, "normal", "anomaly"):
            raise ValueError("prediction must be None, 'normal' or 'anomaly'")


class InferenceBackend(ABC):
    """Base class that every backend (CPU, AI Suite, ...) has to follow."""

    #: Short name of the backend, for example "cpu".
    name: str = "base"
    #: Where it runs, for example "cpu". Used to label the benchmark results correctly.
    device: str = "unknown"

    @abstractmethod
    def load_model(self, model_path) -> None:
        """Load the model and its weights. Raise BackendError if that fails."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """Run the model on one preprocessed image of shape (3, H, W), float32.

        Raise BackendError if the model is not loaded or something goes wrong.
        """

    @property
    def model_name(self) -> str:
        """Name of the model, used in the reports. Backends can override it."""
        return type(self).__name__

    @staticmethod
    def _check_model_path(model_path) -> Path:
        # Helper for backends: make sure the path exists, otherwise raise BackendError.
        p = Path(model_path)
        if not p.exists():
            raise BackendError(f"Model path not found: {p}")
        return p
