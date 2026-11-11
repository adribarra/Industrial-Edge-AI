"""Command-line entry point. Runs the whole thing: dataset, model, inspection, benchmark, reports."""
import argparse
import json
import sys
from pathlib import Path

from app.benchmark import run_benchmark, timed_load
from app.dataset import DatasetError, load_category
from app.inspection import InspectionEngine
from app.preprocessing import PreprocessConfig, PreprocessingError, preprocess_image
from app.reporting import summarize, write_benchmark_json, write_inspection_csv
from inference import BackendError, create_backend


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    p = argparse.ArgumentParser(
        prog="app.main", description="Industrial visual inspection (Phase 1, CPU only)."
    )
    p.add_argument("--dataset", type=Path, default=Path("datasets/mvtec"), help="Dataset root directory")
    p.add_argument("--category", default="screw", help="MVTec AD category")
    p.add_argument("--backend", choices=["cpu"], default="cpu", help="Inference backend")
    p.add_argument("--output", type=Path, default=Path("results"), help="Output directory for reports")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Model directory (default: models/patchcore_resnet18_<category>)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Anomaly threshold (default: value from <model-dir>/threshold.json)")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    p.add_argument("--warmup", type=int, default=3, help="Benchmark warm-up iterations")
    p.add_argument("--size", type=int, default=256, help="Model input width/height")
    return p


def main(argv=None) -> int:
    """Run the full pipeline and return 0 if it worked, 1 if there was an error."""
    a = build_parser().parse_args(argv)
    model_dir = a.model_dir or Path("models") / f"patchcore_resnet18_{a.category}"
    try:
        # 1. list the images
        samples = load_category(a.dataset, a.category, a.split)
        if a.limit:
            samples = samples[: a.limit]

        # 2. get the threshold (from the command line or from the calibration file)
        threshold = a.threshold
        if threshold is None:
            tfile = model_dir / "threshold.json"
            if not tfile.is_file():
                raise BackendError(f"No threshold: {tfile} missing (run scripts/calibrate_threshold.py "
                                   "or pass --threshold)")
            threshold = json.loads(tfile.read_text())["threshold"]

        # 3. load the model
        backend = create_backend(a.backend)
        load_time = timed_load(backend, model_dir)
        cfg = PreprocessConfig(width=a.size, height=a.size)
        engine = InspectionEngine(backend, threshold, cfg)

        # 4. inspect every image
        print(f"Inspecting {len(samples)} {a.split} images of '{a.category}' "
              f"(backend={backend.name}, threshold={threshold:.4f}) ...")
        results = engine.inspect_samples(samples)

        # 5. benchmark (preprocess first so only the model is timed)
        print("Running benchmark ...")
        images = []
        for s in samples:
            try:
                images.append(preprocess_image(s.path, cfg))
            except PreprocessingError:
                pass  # the engine already reported this image as an error row
        bench = run_benchmark(backend, images, warmup=a.warmup, model_load_time_s=load_time)
    except (DatasetError, BackendError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # 6. save the reports and print the summary
    csv_path = write_inspection_csv(results, a.output)
    json_path = write_benchmark_json(
        bench, a.category, a.output,
        {"split": a.split, "threshold": threshold, "input_size": a.size})
    print(summarize(results, bench))
    print(f"Wrote {csv_path}\nWrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
