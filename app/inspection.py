"""The inspection engine: takes the model's score and decides PASS or FAIL.

It only uses the backend interface and the preprocessing code, so it does not care which ML
library is behind the backend.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.dataset import Sample
from app.preprocessing import PreprocessConfig, PreprocessingError, preprocess_image
from inference.base import BackendError, InferenceBackend, Prediction

logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Everything we know about one inspected image (this becomes one row in the CSV)."""

    image_path: Path
    expected_label: Optional[str]      # "normal" / "anomaly" if we know it
    anomaly_score: Optional[float]     # the score from the backend, never modified
    predicted_label: Optional[str]     # "normal" / "anomaly"
    decision: Optional[str]            # "PASS" / "FAIL", or None if something went wrong
    latency_s: Optional[float]         # time spent in backend.predict() only
    threshold: float
    backend: str
    device: str
    error: Optional[str] = None
    prediction: Optional[Prediction] = field(default=None, repr=False)  # raw output, for tracing
    metadata: Dict[str, Any] = field(default_factory=dict)


class InspectionEngine:
    """Runs images through a backend and applies the threshold.

    A score at or above the threshold means anomaly (FAIL), anything below is PASS.
    """

    def __init__(self, backend: InferenceBackend, threshold: float,
                 preprocess_cfg: PreprocessConfig = PreprocessConfig()):
        self.backend = backend
        self.threshold = float(threshold)
        self.preprocess_cfg = preprocess_cfg

    def decide(self, anomaly_score: float) -> str:
        """Compare one score with the threshold and return "PASS" or "FAIL"."""
        return "FAIL" if anomaly_score >= self.threshold else "PASS"

    def inspect_path(self, path, expected_label: Optional[str] = None) -> InspectionResult:
        """Inspect one image file.

        If the image can't be read or the backend fails, the error is stored in the result
        (with no decision) instead of stopping the whole run.
        """
        base = dict(image_path=Path(path), expected_label=expected_label,
                    threshold=self.threshold, backend=self.backend.name,
                    device=self.backend.device)
        try:
            x = preprocess_image(path, self.preprocess_cfg)
            t0 = time.perf_counter()
            pred = self.backend.predict(x)
            latency = time.perf_counter() - t0
        except (PreprocessingError, BackendError) as e:
            logger.warning("Failed to inspect %s: %s", path, e)
            return InspectionResult(anomaly_score=None, predicted_label=None, decision=None,
                                    latency_s=None, error=str(e), **base)
        decision = self.decide(pred.anomaly_score)
        return InspectionResult(
            anomaly_score=pred.anomaly_score,
            predicted_label="anomaly" if decision == "FAIL" else "normal",
            decision=decision, latency_s=latency, prediction=pred,
            metadata={"backend_label": pred.prediction}, **base)

    def inspect_samples(self, samples: Iterable[Sample]) -> List[InspectionResult]:
        """Inspect a list of dataset samples, using their known labels as the expected ones."""
        return [self.inspect_path(s.path, s.label_name) for s in samples]
