"""Measures how fast a backend runs. It works with any backend.

Only backend.predict() is timed, on images that are already preprocessed, using
time.perf_counter() (a monotonic clock). Loading the model is timed separately. Every result
stores the backend and device it was measured on, so a CPU number always says it is a CPU
number.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

from inference.base import InferenceBackend


@dataclass
class BenchmarkResult:
    """All the numbers from one benchmark run."""

    backend: str
    device: str
    model_name: str
    model_load_time_s: float
    warmup_iterations: int
    measured_iterations: int
    total_time_s: float          # wall time of the measured loop only
    mean_latency_s: float
    min_latency_s: float
    max_latency_s: float
    median_latency_s: float
    p95_latency_s: float
    throughput_ips: float        # measured images divided by total_time_s
    latencies_s: List[float]

    def to_dict(self, include_latencies: bool = False) -> Dict:
        """Return the results as a dict. The long list of latencies is left out by default."""
        d = asdict(self)
        if not include_latencies:
            d.pop("latencies_s")
        return d


def timed_load(backend: InferenceBackend, model_path) -> float:
    """Load the model and return how many seconds it took."""
    t0 = time.perf_counter()
    backend.load_model(model_path)
    return time.perf_counter() - t0


def run_benchmark(backend: InferenceBackend, images: Sequence[np.ndarray],
                  warmup: int = 1, model_load_time_s: float = float("nan")) -> BenchmarkResult:
    """Time backend.predict() on each image (the model must already be loaded).

    The first `warmup` runs are done but not counted, because the first calls are usually
    slower. After that every image is run once and timed.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for i in range(warmup):
        backend.predict(images[i % len(images)])

    latencies: List[float] = []
    loop_start = time.perf_counter()
    for img in images:
        t0 = time.perf_counter()
        backend.predict(img)
        latencies.append(time.perf_counter() - t0)
    total = time.perf_counter() - loop_start

    a = np.asarray(latencies)
    return BenchmarkResult(
        backend=backend.name, device=backend.device, model_name=backend.model_name,
        model_load_time_s=model_load_time_s, warmup_iterations=warmup,
        measured_iterations=len(a), total_time_s=total,
        mean_latency_s=float(a.mean()), min_latency_s=float(a.min()),
        max_latency_s=float(a.max()), median_latency_s=float(np.median(a)),
        p95_latency_s=float(np.percentile(a, 95)),
        throughput_ips=len(a) / total, latencies_s=latencies,
    )
