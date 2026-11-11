import numpy as np
import pytest
from PIL import Image

from app.dataset import Sample
from app.inspection import InspectionEngine
from app.postprocessing import calibrate_threshold
from app.preprocessing import PreprocessConfig
from inference.base import BackendError, InferenceBackend, Prediction

CFG = PreprocessConfig(width=8, height=8, mean=(0, 0, 0), std=(1, 1, 1))


class BrightnessBackend(InferenceBackend):
    """Test-only: score = mean pixel value in [0,1]. Not a model."""

    name, device = "fake", "cpu"

    def load_model(self, model_path):
        pass

    def predict(self, image):
        return Prediction(float(image.mean()))


class FailingBackend(BrightnessBackend):
    """Fake backend that always fails, to test error handling."""
    def predict(self, image):
        raise BackendError("boom")


def _png(path, value):
    """Save a small image filled with one gray value."""
    Image.fromarray(np.full((10, 10, 3), value, np.uint8)).save(path)
    return path


def test_threshold_decision(tmp_path):
    """Dark image passes, bright image fails, and the scores are kept as they are."""
    dark, bright = _png(tmp_path / "d.png", 51), _png(tmp_path / "b.png", 204)  # 0.2 / 0.8
    eng = InspectionEngine(BrightnessBackend(), threshold=0.5, preprocess_cfg=CFG)
    d, b = eng.inspect_path(dark, "normal"), eng.inspect_path(bright, "anomaly")
    assert (d.decision, d.predicted_label) == ("PASS", "normal")
    assert (b.decision, b.predicted_label) == ("FAIL", "anomaly")
    assert d.anomaly_score == pytest.approx(0.2) and b.anomaly_score == pytest.approx(0.8)


def test_threshold_is_configurable_and_inclusive(tmp_path):
    """A score equal to the threshold fails, and a higher threshold lets it pass."""
    img = _png(tmp_path / "x.png", 255)  # score 1.0
    assert InspectionEngine(BrightnessBackend(), 1.0, CFG).inspect_path(img).decision == "FAIL"
    assert InspectionEngine(BrightnessBackend(), 1.5, CFG).inspect_path(img).decision == "PASS"


def test_traceability_and_metadata(tmp_path):
    """The result keeps the path, labels, original score, latency and backend info."""
    img = _png(tmp_path / "x.png", 100)
    r = InspectionEngine(BrightnessBackend(), 0.9, CFG).inspect_path(img, "normal")
    assert r.image_path == img and r.expected_label == "normal"
    assert r.prediction.anomaly_score == r.anomaly_score  # original score preserved
    assert r.latency_s >= 0 and r.threshold == 0.9
    assert (r.backend, r.device) == ("fake", "cpu")


def test_errors_are_recorded_not_raised(tmp_path):
    """Missing images and backend errors end up in the result, not as a crash."""
    eng = InspectionEngine(BrightnessBackend(), 0.5, CFG)
    r = eng.inspect_path(tmp_path / "missing.png")
    assert r.decision is None and r.error
    r2 = InspectionEngine(FailingBackend(), 0.5, CFG).inspect_path(_png(tmp_path / "a.png", 1))
    assert r2.decision is None and "boom" in r2.error


def test_inspect_samples(tmp_path):
    """A list of samples gives one decision each, with the expected labels."""
    s = [Sample(_png(tmp_path / "a.png", 0), "test", "good", 0),
         Sample(_png(tmp_path / "b.png", 255), "test", "scratch", 1)]
    res = InspectionEngine(BrightnessBackend(), 0.5, CFG).inspect_samples(s)
    assert [r.decision for r in res] == ["PASS", "FAIL"]
    assert [r.expected_label for r in res] == ["normal", "anomaly"]


def test_calibrate_threshold():
    """The threshold calculation (percentile and margin) and its input checks."""
    scores = [1.0, 2.0, 3.0, 4.0]
    assert calibrate_threshold(scores) == 4.0
    assert calibrate_threshold(scores, 50) == pytest.approx(2.5)
    assert calibrate_threshold(scores, margin=0.1) == pytest.approx(4.4)
    with pytest.raises(ValueError):
        calibrate_threshold([])
    with pytest.raises(ValueError):
        calibrate_threshold([1.0], percentile=101)
