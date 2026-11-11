"""CPU backend tests.

Mechanics tests use a randomly initialised ResNet18 saved to a temp dir, so
they need no downloads. They check shapes/behaviour, NOT detection quality.
The integration test uses the real model and is skipped if it is absent.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision

from app.dataset import load_category
from app.preprocessing import preprocess_image
from inference.base import BackendError
from inference.cpu_backend import WEIGHTS_FILE, CPUBackend

REAL_MODEL = Path("models/patchcore_resnet18_screw")
REAL_DATA = Path("datasets/mvtec")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Build a small model folder with random weights, just to test the code (not real detection)."""
    d = tmp_path_factory.mktemp("model")
    torch.manual_seed(0)
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), d / WEIGHTS_FILE)
    rng = np.random.default_rng(0)
    normal = [rng.normal(size=(3, 64, 64)).astype(np.float32) for _ in range(3)]
    b = CPUBackend()
    assert b.fit(d, normal, keep_fraction=0.5) > 0
    b.save(d)
    return d


def test_predict_shapes_and_range(model_dir):
    """The score is a valid number and the anomaly map has the expected shape."""
    b = CPUBackend()
    b.load_model(model_dir)
    x = np.random.default_rng(1).normal(size=(3, 64, 64)).astype(np.float32)
    p = b.predict(x)
    assert np.isfinite(p.anomaly_score) and p.anomaly_score >= 0
    assert p.anomaly_map.shape == (8, 8) and p.anomaly_map.dtype == np.float32
    assert p.anomaly_score == pytest.approx(float(p.anomaly_map.max()))
    assert (b.name, b.device) == ("cpu", "cpu")


def test_deterministic(model_dir):
    """Running the same image twice gives the same score."""
    b = CPUBackend()
    b.load_model(model_dir)
    x = np.ones((3, 64, 64), np.float32)
    assert b.predict(x).anomaly_score == b.predict(x).anomaly_score


def test_training_like_image_scores_lower_than_outlier(model_dir):
    """An image from the memory scores lower than an extreme outlier."""
    b = CPUBackend()
    b.load_model(model_dir)
    rng = np.random.default_rng(0)
    seen = rng.normal(size=(3, 64, 64)).astype(np.float32)  # same as a bank image
    outlier = (seen * 0 + 25).astype(np.float32)
    assert b.predict(seen).anomaly_score < b.predict(outlier).anomaly_score


def test_errors(tmp_path, model_dir):
    """Predicting before loading, bad paths, wrong dtype and wrong shape all raise BackendError."""
    b = CPUBackend()
    with pytest.raises(BackendError):
        b.predict(np.zeros((3, 64, 64), np.float32))  # not loaded
    with pytest.raises(BackendError):
        b.load_model(tmp_path / "missing")
    with pytest.raises(BackendError):
        b.load_model(tmp_path)  # exists but no weights
    b.load_model(model_dir)
    with pytest.raises(BackendError):
        b.predict(np.zeros((3, 64, 64), np.float64))  # wrong dtype
    with pytest.raises(BackendError):
        b.predict(np.zeros((64, 64), np.float32))  # wrong shape


@pytest.mark.skipif(
    not (REAL_MODEL / "memory_bank.pt").exists() or not REAL_DATA.exists(),
    reason="real model/dataset not available",
)
def test_real_image_end_to_end():
    """With the real model and dataset, one real image goes through and gives a valid result."""
    b = CPUBackend()
    b.load_model(REAL_MODEL)
    sample = load_category(REAL_DATA, "screw", "test")[0]
    p = b.predict(preprocess_image(sample.path))
    assert np.isfinite(p.anomaly_score) and p.anomaly_map.ndim == 2
