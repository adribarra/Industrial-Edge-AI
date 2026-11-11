"""Writes the results: inspection_results.csv, benchmark.json and the text summary for the terminal."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from app.benchmark import BenchmarkResult
from app.inspection import InspectionResult

CSV_COLUMNS = [
    "image_path", "expected_label", "predicted_label", "anomaly_score",
    "decision", "latency_s", "error",
]


def results_to_dataframe(results: Sequence[InspectionResult]) -> pd.DataFrame:
    """Put the inspection results in a table with one row per image."""
    rows = [{
        "image_path": str(r.image_path),
        "expected_label": r.expected_label,
        "predicted_label": r.predicted_label,
        "anomaly_score": r.anomaly_score,
        "decision": r.decision,
        "latency_s": r.latency_s,
        "error": r.error,
    } for r in results]
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def write_inspection_csv(results: Sequence[InspectionResult], output_dir) -> Path:
    """Save inspection_results.csv in the output folder (created if needed) and return its path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "inspection_results.csv"
    results_to_dataframe(results).to_csv(path, index=False)
    return path


def write_benchmark_json(bench: BenchmarkResult, dataset_category: str, output_dir,
                         extra_config: Optional[Dict[str, Any]] = None) -> Path:
    """Save benchmark.json in the output folder and return its path.

    extra_config is added to the measurement settings (for example the threshold used).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "benchmark.json"
    doc = {
        "model_name": bench.model_name,
        "backend": bench.backend,
        "execution_device": bench.device,
        "dataset_category": dataset_category,
        "processed_images": bench.measured_iterations,
        "average_latency_s": bench.mean_latency_s,
        "median_latency_s": bench.median_latency_s,
        "p95_latency_s": bench.p95_latency_s,
        "min_latency_s": bench.min_latency_s,
        "max_latency_s": bench.max_latency_s,
        "throughput_images_per_s": bench.throughput_ips,
        "total_execution_time_s": bench.total_time_s,
        "model_load_time_s": bench.model_load_time_s,
        "measurement_config": {
            "warmup_iterations": bench.warmup_iterations,
            "measured_iterations": bench.measured_iterations,
            "timer": "time.perf_counter (monotonic)",
            "measures": "backend.predict() on preprocessed images; excludes model load and preprocessing",
            **(extra_config or {}),
        },
        "note": f"Results measured on device '{bench.device}' only; not comparable to other hardware.",
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def summarize(results: Sequence[InspectionResult], bench: Optional[BenchmarkResult] = None) -> str:
    """Build the readable summary printed at the end of a run.

    It shows PASS/FAIL counts, how many defects were found or missed (if the expected
    labels are known), and the benchmark numbers if there are any.
    """
    ok = [r for r in results if r.decision is not None]
    failed = len(results) - len(ok)
    lines = ["=" * 56, "Inspection summary", "=" * 56,
             f"Images processed : {len(ok)} ({failed} errors)",
             f"PASS / FAIL      : {sum(r.decision == 'PASS' for r in ok)} / "
             f"{sum(r.decision == 'FAIL' for r in ok)}"]
    labeled = [r for r in ok if r.expected_label]
    if labeled:
        tp = sum(r.expected_label == "anomaly" and r.decision == "FAIL" for r in labeled)
        fn = sum(r.expected_label == "anomaly" and r.decision == "PASS" for r in labeled)
        fp = sum(r.expected_label == "normal" and r.decision == "FAIL" for r in labeled)
        tn = sum(r.expected_label == "normal" and r.decision == "PASS" for r in labeled)
        lines += [f"Anomalous found  : {tp}/{tp + fn} (missed {fn})",
                  f"False alarms     : {fp}/{fp + tn}",
                  f"Accuracy         : {(tp + tn) / len(labeled):.3f}"]
    if bench:
        lines += ["-" * 56,
                  f"Benchmark ({bench.backend}, device={bench.device}, {bench.model_name})",
                  f"  Load time      : {bench.model_load_time_s * 1e3:.1f} ms",
                  f"  Latency mean   : {bench.mean_latency_s * 1e3:.1f} ms",
                  f"  Latency median : {bench.median_latency_s * 1e3:.1f} ms",
                  f"  Latency P95    : {bench.p95_latency_s * 1e3:.1f} ms",
                  f"  Throughput     : {bench.throughput_ips:.2f} images/s",
                  f"  Warm-up/measured: {bench.warmup_iterations}/{bench.measured_iterations}",
                  f"  NOTE: {bench.device.upper()} measurement only."]
    lines.append("=" * 56)
    return "\n".join(lines)
