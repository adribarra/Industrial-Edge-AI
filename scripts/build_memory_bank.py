"""Downloads the ResNet18 weights and builds the model's memory from good training images.

Only train/good images are used, so the test images have no part in building the model.
About 10% of the good images are set aside for choosing the threshold later.
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset import load_category  # noqa: E402
from app.preprocessing import PreprocessConfig, preprocess_image  # noqa: E402
from inference.cpu_backend import WEIGHTS_FILE, CPUBackend  # noqa: E402

WEIGHTS_URL = "https://download.pytorch.org/models/" + WEIGHTS_FILE


def main() -> int:
    """Download the weights (if needed), split the images, build and save the memory bank."""
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("datasets/mvtec"))
    p.add_argument("--category", default="screw")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--keep-fraction", type=float, default=0.1)
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="share of normal train images held out for threshold calibration")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    model_dir = a.model_dir or Path("models") / f"patchcore_resnet18_{a.category}"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = model_dir / WEIGHTS_FILE
    if not weights.is_file():
        print(f"Downloading ResNet18 weights (~47 MB) from {WEIGHTS_URL}")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    cfg = PreprocessConfig(width=a.size, height=a.size)
    samples = [s for s in load_category(a.dataset, a.category, "train") if s.label == 0]
    random.Random(a.seed).shuffle(samples)
    n_val = int(len(samples) * a.val_fraction)
    val, fit = samples[:n_val], samples[n_val:]
    (model_dir / "validation_images.json").write_text(
        json.dumps(sorted(str(s.path) for s in val), indent=1))
    print(f"Building memory bank from {len(fit)} normal train images "
          f"({len(val)} held out for validation) ...")
    backend = CPUBackend()
    n = backend.fit(model_dir, (preprocess_image(s.path, cfg) for s in fit), a.keep_fraction)
    out = backend.save(model_dir)
    print(f"Saved {n} patches to {out} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
