import numpy as np
import pytest
from PIL import Image

from app.main import build_parser, main


def test_parser_defaults():
    """Default command-line options are the expected ones."""
    a = build_parser().parse_args([])
    assert a.category == "screw" and a.backend == "cpu" and a.threshold is None


def test_missing_dataset_returns_error(tmp_path, capsys):
    """A missing dataset folder gives an error message and exit code 1."""
    rc = main(["--dataset", str(tmp_path), "--output", str(tmp_path / "o")])
    assert rc == 1 and "Dataset folder not found" in capsys.readouterr().err


def test_missing_threshold_returns_error(tmp_path, capsys):
    """A missing threshold file gives an error message and exit code 1."""
    img = tmp_path / "ds/screw/test/good/0.png"
    img.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(img)
    rc = main(["--dataset", str(tmp_path / "ds"), "--model-dir", str(tmp_path / "nomodel")])
    assert rc == 1 and "No threshold" in capsys.readouterr().err
