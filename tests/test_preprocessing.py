import numpy as np
import pytest
from PIL import Image

from app.preprocessing import (
    PreprocessConfig,
    PreprocessingError,
    preprocess_array,
    preprocess_image,
)


def _rgb(h=40, w=60, value=None):
    """Make a fake RGB image, random or filled with one value."""
    if value is None:
        return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_shape_and_dtype_default():
    """Default output is float32 with shape (3, 256, 256)."""
    out = preprocess_array(_rgb())
    assert out.shape == (3, 256, 256)
    assert out.dtype == np.float32


def test_configurable_size():
    """The output size follows the width/height setting."""
    out = preprocess_array(_rgb(), PreprocessConfig(width=64, height=32))
    assert out.shape == (3, 32, 64)


def test_normalization_values():
    """White and black pixels give exactly +1 and -1 with mean/std of 0.5."""
    cfg = PreprocessConfig(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    assert np.allclose(preprocess_array(_rgb(value=255), cfg), 1.0)
    assert np.allclose(preprocess_array(_rgb(value=0), cfg), -1.0)


def test_imagenet_channel_order():
    """Red ends up in the first channel (RGB order, not BGR)."""
    img = np.zeros((10, 10, 3), np.uint8)
    img[..., 0] = 255  # pure red in RGB
    out = preprocess_array(img, PreprocessConfig(width=4, height=4))
    assert np.allclose(out[0], (1 - 0.485) / 0.229)
    assert np.allclose(out[1], (0 - 0.456) / 0.224)


def test_deterministic():
    """The same image always gives the same output."""
    img = _rgb()
    assert np.array_equal(preprocess_array(img), preprocess_array(img))


def test_from_file_grayscale_and_rgb(tmp_path):
    """Both grayscale and color files load as 3-channel images."""
    Image.fromarray(np.zeros((20, 30), np.uint8)).save(tmp_path / "g.png")
    Image.fromarray(_rgb()).save(tmp_path / "c.png")
    assert preprocess_image(tmp_path / "g.png").shape == (3, 256, 256)
    assert preprocess_image(tmp_path / "c.png").shape == (3, 256, 256)


def test_missing_and_corrupt_file(tmp_path):
    """A missing or broken file raises PreprocessingError."""
    with pytest.raises(PreprocessingError):
        preprocess_image(tmp_path / "nope.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(PreprocessingError):
        preprocess_image(bad)


@pytest.mark.parametrize(
    "arr",
    [np.zeros((5, 5), np.uint8), np.zeros((5, 5, 4), np.uint8), np.zeros((5, 5, 3), np.float32)],
)
def test_invalid_arrays(arr):
    """Wrong shape or wrong data type is rejected."""
    with pytest.raises(PreprocessingError):
        preprocess_array(arr)


def test_invalid_config():
    """Bad settings (zero width, zero std) are rejected."""
    with pytest.raises(ValueError):
        PreprocessConfig(width=0)
    with pytest.raises(ValueError):
        PreprocessConfig(std=(1, 0, 1))
