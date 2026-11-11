import json
from pathlib import Path

import pandas as pd

from app.benchmark import BenchmarkResult
from app.inspection import InspectionResult
from app.reporting import (CSV_COLUMNS, summarize, write_benchmark_json,
                           write_inspection_csv)


def _res(name, expected, score, decision, error=None):
    """Make a fake inspection result for the tests."""
    return InspectionResult(
        image_path=Path(name), expected_label=expected, anomaly_score=score,
        predicted_label=None if decision is None else ("anomaly" if decision == "FAIL" else "normal"),
        decision=decision, latency_s=0.01 if decision else None, threshold=1.0,
        backend="cpu", device="cpu", error=error)


RESULTS = [
    _res("a.png", "normal", 0.5, "PASS"),
    _res("b.png", "anomaly", 2.0, "FAIL"),
    _res("c.png", "anomaly", 0.7, "PASS"),
    _res("d.png", "normal", 1.5, "FAIL"),
    _res("e.png", None, None, None, error="cannot decode"),
]


def _bench():
    """Make a fake benchmark result with fixed numbers."""
    return BenchmarkResult(
        backend="cpu", device="cpu", model_name="m", model_load_time_s=0.2,
        warmup_iterations=2, measured_iterations=4, total_time_s=1.0,
        mean_latency_s=0.25, min_latency_s=0.2, max_latency_s=0.3,
        median_latency_s=0.24, p95_latency_s=0.29, throughput_ips=4.0,
        latencies_s=[0.2, 0.25, 0.25, 0.3])


def test_csv_content(tmp_path):
    """The CSV has the right columns and values, including a row with an error."""
    p = write_inspection_csv(RESULTS, tmp_path / "out")  # creates directory
    assert p.name == "inspection_results.csv"
    df = pd.read_csv(p)
    assert list(df.columns) == CSV_COLUMNS and len(df) == 5
    assert df.loc[1, "decision"] == "FAIL" and df.loc[1, "anomaly_score"] == 2.0
    assert df.loc[4, "error"] == "cannot decode" and pd.isna(df.loc[4, "decision"])


def test_benchmark_json(tmp_path):
    """The JSON has all the required fields and never mentions FPGA."""
    p = write_benchmark_json(_bench(), "screw", tmp_path, {"threshold": 1.0})
    d = json.loads(p.read_text())
    assert p.name == "benchmark.json"
    for k in ["model_name", "backend", "execution_device", "dataset_category",
              "processed_images", "average_latency_s", "median_latency_s",
              "p95_latency_s", "throughput_images_per_s", "total_execution_time_s",
              "measurement_config"]:
        assert k in d
    assert d["execution_device"] == "cpu" and d["dataset_category"] == "screw"
    assert d["measurement_config"]["warmup_iterations"] == 2
    assert d["measurement_config"]["threshold"] == 1.0
    assert "fpga" not in json.dumps(d).lower()


def test_summary_counts_and_cpu_label():
    """The terminal summary shows the right counts and says CPU only."""
    s = summarize(RESULTS, _bench())
    assert "Images processed : 4 (1 errors)" in s
    assert "PASS / FAIL      : 2 / 2" in s
    assert "Anomalous found  : 1/2 (missed 1)" in s
    assert "False alarms     : 1/2" in s
    assert "CPU measurement only" in s


def test_summary_without_labels_or_benchmark():
    """The summary still works without expected labels or a benchmark."""
    s = summarize([_res("x.png", None, 0.1, "PASS")])
    assert "Accuracy" not in s and "Benchmark" not in s
