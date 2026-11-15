import importlib.util
from pathlib import Path

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_comparisons", ROOT / "scripts/build_comparisons.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_arrangement_is_pixel_exact_and_resume_skips_good_file(tmp_path):
    left, right = Image.new("RGB", (224, 224), (17, 31, 97)), Image.new("RGB", (224, 224), (2, 137, 73))
    target = tmp_path / "pair.png"
    assert module.write_comparison(target, left, right)
    before, mtime = target.read_bytes(), target.stat().st_mtime_ns
    assert module.exact_halves(target, left, right)
    assert not module.exact_halves(target, right, left)
    assert not module.write_comparison(target, left, right)
    assert target.read_bytes() == before and target.stat().st_mtime_ns == mtime


def test_corrupt_comparison_and_unfinished_file_are_repaired(tmp_path):
    left, right = Image.new("RGB", (224, 224), "white"), Image.new("RGB", (224, 224), "black")
    target = tmp_path / "pair.png"
    target.write_bytes(b"unfinished invalid image")
    (tmp_path / "pair.png.part").write_bytes(b"interrupted output")
    assert module.write_comparison(target, left, right)
    assert module.exact_halves(target, left, right)
    assert not (tmp_path / "pair.png.part").exists()


def test_no_silent_resize_or_mode_conversion(tmp_path):
    for left in (Image.new("RGB", (225, 224)), Image.new("L", (224, 224))):
        with pytest.raises(ValueError):
            module.write_comparison(tmp_path / "bad.png", left, Image.new("RGB", (224, 224)))
    assert not (tmp_path / "bad.png").exists()


@pytest.mark.parametrize("pair", [
    {"split": "../", "pair_id": "ffpair_" + "a" * 20},
    {"split": "train", "pair_id": "../source"},
    {"split": "invalid", "pair_id": "ffpair_" + "a" * 20},
])
def test_reject_unsafe_output_names(pair):
    with pytest.raises(ValueError):
        module.relative_path(pair)
