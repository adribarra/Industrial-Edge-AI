import numpy as np
import pytest
from PIL import Image

from app.dataset import DatasetError, load_category


def _img(path):
    """Save a tiny black image at this path (creates the folders too)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def mvtec(tmp_path):
    """Build a small fake MVTec folder: good and defective images, one mask missing, one text file."""
    cat = tmp_path / "screw"
    _img(cat / "train/good/000.png")
    _img(cat / "train/good/001.png")
    _img(cat / "test/good/000.png")
    _img(cat / "test/scratch_head/000.png")
    _img(cat / "test/scratch_head/001.png")
    _img(cat / "ground_truth/scratch_head/000_mask.png")  # 001 mask missing
    (cat / "test/good/notes.txt").write_text("not an image")
    return tmp_path


def test_test_split_labels(mvtec):
    """The test split has the right number of images and labels."""
    s = load_category(mvtec, "screw", "test")
    assert len(s) == 3
    assert sorted(x.label for x in s) == [0, 1, 1]
    assert {x.defect_type for x in s} == {"good", "scratch_head"}


def test_masks_only_for_anomalies(mvtec):
    """Only defective images get a mask, and a missing mask does not crash."""
    s = {(x.defect_type, x.path.stem): x for x in load_category(mvtec, "screw")}
    assert s[("scratch_head", "000")].mask_path is not None
    assert s[("scratch_head", "001")].mask_path is None  # missing, handled
    assert s[("good", "000")].mask_path is None


def test_train_split_all_normal(mvtec):
    """The train split only has good images."""
    s = load_category(mvtec, "screw", "train")
    assert len(s) == 2 and all(x.label == 0 for x in s)


def test_missing_category(mvtec):
    """A category that does not exist raises DatasetError."""
    with pytest.raises(DatasetError):
        load_category(mvtec, "hazelnut")


def test_invalid_split(mvtec):
    """A split name other than train/test raises ValueError."""
    with pytest.raises(ValueError):
        load_category(mvtec, "screw", "val")
