"""Chooses the PASS/FAIL threshold using the good validation images that were set aside.

The test images are never used. The result is saved as threshold.json next to the model.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.postprocessing import calibrate_threshold  # noqa: E402
from app.preprocessing import PreprocessConfig, preprocess_image  # noqa: E402
from inference.cpu_backend import CPUBackend  # noqa: E402


def main() -> int:
    """Score the validation images, pick the threshold and save it."""
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=Path("models/patchcore_resnet18_screw"))
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--percentile", type=float, default=100.0)
    p.add_argument("--margin", type=float, default=0.0)
    a = p.parse_args()

    val = json.loads((a.model_dir / "validation_images.json").read_text())
    backend = CPUBackend()
    backend.load_model(a.model_dir)
    cfg = PreprocessConfig(width=a.size, height=a.size)
    scores = [backend.predict(preprocess_image(v, cfg)).anomaly_score for v in val]
    thr = calibrate_threshold(scores, a.percentile, a.margin)
    out = a.model_dir / "threshold.json"
    out.write_text(json.dumps({
        "threshold": thr, "percentile": a.percentile, "margin": a.margin,
        "n_validation_images": len(scores),
        "validation_score_min": min(scores), "validation_score_max": max(scores)}, indent=1))
    print(f"{len(scores)} validation images, scores {min(scores):.3f}..{max(scores):.3f}")
    print(f"threshold = {thr:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
