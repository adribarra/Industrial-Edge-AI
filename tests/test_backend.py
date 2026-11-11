import numpy as np
import pytest

from inference.ai_suite_backend import AISuiteBackend
from inference.base import BackendError, InferenceBackend, Prediction


class MeanBackend(InferenceBackend):
    """Test-only deterministic backend (score = mean pixel value). Not a model."""

    name = "test"
    device = "cpu"

    def __init__(self):
        self.loaded = False

    def load_model(self, model_path):
        self._check_model_path(model_path)
        self.loaded = True

    def predict(self, image):
        if not self.loaded:
            raise BackendError("model not loaded")
        return Prediction(anomaly_score=float(image.mean()))


def test_cannot_instantiate_abstract():
    """The abstract base class cannot be used directly."""
    with pytest.raises(TypeError):
        InferenceBackend()


def test_backend_contract(tmp_path):
    """A backend must refuse to predict before loading and refuse a bad model path."""
    b = MeanBackend()
    with pytest.raises(BackendError):
        b.predict(np.zeros((3, 4, 4), np.float32))  # not loaded
    with pytest.raises(BackendError):
        b.load_model(tmp_path / "missing")
    b.load_model(tmp_path)
    p = b.predict(np.ones((3, 4, 4), np.float32))
    assert p.anomaly_score == 1.0 and p.prediction is None and p.anomaly_map is None


def test_prediction_validation():
    """Prediction rejects NaN scores and unknown labels."""
    assert Prediction(0.5, "anomaly").anomaly_score == 0.5
    with pytest.raises(ValueError):
        Prediction(float("nan"))
    with pytest.raises(ValueError):
        Prediction(0.1, "maybe")


def test_ai_suite_is_stub_only():
    """The AI Suite placeholder fails clearly instead of pretending to work."""
    b = AISuiteBackend()
    assert b.device == "fpga"
    with pytest.raises(BackendError):
        b.load_model("x")
    with pytest.raises(BackendError):
        b.predict(np.zeros((3, 4, 4), np.float32))
