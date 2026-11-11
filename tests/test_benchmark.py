import time

import numpy as np
import pytest

from app.benchmark import run_benchmark, timed_load
from inference.base import InferenceBackend, Prediction


class SleepBackend(InferenceBackend):
    """Test-only backend with a controllable delay. Not a model."""

    name, device = "fake", "cpu"

    def __init__(self, delays):
        self.delays, self.calls = list(delays), 0

    def load_model(self, model_path):
        time.sleep(0.05)

    def predict(self, image):
        time.sleep(self.delays[self.calls % len(self.delays)])
        self.calls += 1
        return Prediction(0.0)


IMG = np.zeros((3, 4, 4), np.float32)


def test_counts_and_warmup_excluded():
    """Warm-up runs happen but are not counted in the measurements."""
    b = SleepBackend([0.001])
    r = run_benchmark(b, [IMG] * 5, warmup=2)
    assert r.measured_iterations == 5 and r.warmup_iterations == 2
    assert b.calls == 7 and len(r.latencies_s) == 5


def test_statistics_consistent():
    """min <= median <= P95 <= max, and throughput matches the total time."""
    r = run_benchmark(SleepBackend([0.002, 0.004, 0.006]), [IMG] * 6, warmup=0)
    assert r.min_latency_s <= r.median_latency_s <= r.p95_latency_s <= r.max_latency_s
    assert r.min_latency_s >= 0.002 and r.max_latency_s >= 0.006
    assert r.mean_latency_s == pytest.approx(np.mean(r.latencies_s))
    assert r.throughput_ips == pytest.approx(6 / r.total_time_s)
    assert r.total_time_s >= sum(r.latencies_s) - 1e-9


def test_backend_and_device_recorded():
    """The result says which backend and device it was measured on."""
    r = run_benchmark(SleepBackend([0]), [IMG], warmup=0)
    assert (r.backend, r.device) == ("fake", "cpu")
    assert "latencies_s" not in r.to_dict() and "latencies_s" in r.to_dict(True)


def test_load_time_separate():
    """Model load time is not mixed into the inference time."""
    b = SleepBackend([0])
    load = timed_load(b, "ignored")
    assert load >= 0.05
    r = run_benchmark(b, [IMG] * 2, warmup=0, model_load_time_s=load)
    assert r.model_load_time_s == load and r.total_time_s < load


def test_invalid_arguments():
    """Empty image list or negative warm-up is rejected."""
    with pytest.raises(ValueError):
        run_benchmark(SleepBackend([0]), [])
    with pytest.raises(ValueError):
        run_benchmark(SleepBackend([0]), [IMG], warmup=-1)
