from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from matching import capped_allocation, match, pair_pool  # noqa: E402
from pipeline import HashIndex, annotation, metrics, verify_annotation  # noqa: E402
from storage import Store, atomic_bytes, digest, sha, writer_lock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "config/curation.json").read_text())


def row(i, race="Black", gender="Female", age="20-29"):
    return {"image_id": f"{race}_{gender}_{i:03}", "source_race": race, "source_gender": gender,
            "age_band": age, "expression_class": "neutral", "expression_label": "neutral",
            "expression_confidence": .7 + (i % 10) * .01, "contrast": .15 + i * .001,
            "log_edge_energy": -4 + i * .005, "image_sha256": "abc"}


def test_apportionment():
    assert capped_allocation(5, {"a": 1, "b": 8}) == {"a": 1, "b": 4}
    with pytest.raises(ValueError):
        capped_allocation(10, {"a": 1})


def test_matching_balance_and_order_independence(config):
    config["target_pairs"] = 20
    rows = [row(i, race, gender) for race in config["races"] for gender in config["genders"] for i in range(12)]
    first = match(rows, config)
    second = match(list(reversed(rows)), config)
    assert first == second
    pairs, _, report = first
    assert report["pairs_by_split"] == {"train": 12, "development": 4, "test": 4}
    assert report["pairs_by_gender"] == {"Female": 10, "Male": 10}
    assert len({p[k] for p in pairs for k in ("black_image_id", "white_image_id")}) == 40
    for p in pairs:
        assert abs(p["contrast_difference_sd"]) < 1e-8


def test_calipers_prevent_bad_pairs(config):
    a, b = row(1), row(2, "White")
    b["expression_confidence"] = .99
    assert pair_pool([a], [b], config, {"contrast": 1, "log_edge_energy": 1}) == []
    b["expression_confidence"] = .71
    b["contrast"] = 4
    assert pair_pool([a], [b], config, {"contrast": 1, "log_edge_energy": 1}) == []


def test_cardinality_precedes_cost(config):
    black = [row(0), row(1)]
    white = [row(0, "White"), row(1, "White")]
    black[0]["expression_confidence"] = .6
    black[1]["expression_confidence"] = .75
    white[0]["expression_confidence"] = .70
    white[1]["expression_confidence"] = .85
    pairs = pair_pool(black, white, config, {"contrast": 1, "log_edge_energy": 1})
    assert len(pairs) == 2
    assert all(p["confidence_abs_difference"] <= .15 + 1e-12 for p in pairs)


def test_shortfall_is_not_filled(config):
    config["target_pairs"] = 100
    rows = [row(i, race, gender) for race in config["races"] for gender in config["genders"] for i in range(5)]
    _, _, report = match(rows, config)
    assert report["selected_pairs"] == 10 and report["shortfall_pairs"] == 90


def test_detailed_expression_matching(config):
    rows = [row(i, race, gender) for race in config["races"] for gender in config["genders"] for i in range(3)]
    for r in rows:
        r["expression_class"] = "negative"
        r["expression_label"] = "sad" if r["source_race"] == "Black" else "angry"
    pairs, _, _ = match(rows, config)
    assert pairs == []


def test_classifier_uncertainty_is_explicit(config):
    r = row(0)
    a = annotation(r, "happy", .59, config, "test")
    assert a["expression_class"] == "uncertain"
    verify_annotation(r, a, config)
    a["image_sha256"] = "drift"
    with pytest.raises(ValueError):
        verify_annotation(r, a, config)


def test_hash_exact_and_near(config):
    index = HashIndex()
    r = metrics(Image.fromarray(np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)))
    index.add(r)
    assert index.matches(r, config) == r
    near = dict(r, pixel_sha256="new", perceptual_hash=f"{int(r['perceptual_hash'], 16) ^ 1:016x}")
    assert index.matches(near, config) == r


def test_checkpoint_roundtrip_drift_and_corruption(tmp_path):
    p = tmp_path / "checkpoint.sqlite"
    s = Store(p, {"version": 1})
    s.put("screen", [{"image_id": "a", "value": 3}])
    s.backup(tmp_path / "backup.sqlite")
    s.close()
    s = Store(p, {"version": 1})
    assert s.get("screen")["a"]["value"] == 3
    s.close()
    with pytest.raises(ValueError, match="contract changed"):
        Store(p, {"version": 2})
    db = sqlite3.connect(p)
    db.execute("UPDATE records SET payload='{}'")
    db.commit()
    db.close()
    s = Store(p, {"version": 1})
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        s.get("screen")
    s.close()
    backup = Store(tmp_path / "backup.sqlite", {"version": 1})
    assert backup.get("screen")["a"]["value"] == 3
    backup.close()


def test_process_killed_mid_transaction_then_resume(tmp_path):
    path = tmp_path / "state.sqlite"
    s = Store(path, {"version": 1})
    s.put("screen", [{"image_id": "completed", "n": 1}])
    s.close()
    code = """
import os, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
db.execute('BEGIN IMMEDIATE')
db.execute("INSERT INTO records VALUES ('screen','unfinished','{}','bad')")
os._exit(79)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)])
    assert result.returncode == 79
    s = Store(path, {"version": 1})
    assert set(s.get("screen")) == {"completed"}
    s.put("screen", [{"image_id": "unfinished", "n": 2}])
    actual = s.get("screen")
    s.close()
    uninterrupted = Store(tmp_path / "whole.sqlite", {"version": 1})
    uninterrupted.put("screen", [{"image_id": "completed", "n": 1}, {"image_id": "unfinished", "n": 2}])
    assert uninterrupted.get("screen") == actual
    uninterrupted.close()


def test_interrupted_atomic_write_preserves_old_then_recovers(tmp_path):
    path = tmp_path / "output.json"
    atomic_bytes(path, b"old-complete")
    code = """
import os, sys
with open(sys.argv[1]+'.part','wb') as f:
    f.write(b'incomplete')
    f.flush()
    os.fsync(f.fileno())
os._exit(79)
"""
    assert subprocess.run([sys.executable, "-c", code, str(path)]).returncode == 79
    assert path.read_bytes() == b"old-complete"
    atomic_bytes(path, b"new-complete")
    assert sha(path) == digest(b"new-complete")
    assert not path.with_name(path.name + ".part").exists()


def test_lock_prevents_second_writer_and_recovers(tmp_path):
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); from storage import writer_lock; "
    code += f"\nwith writer_lock({str(tmp_path)!r}):\n print('acquired')"
    with writer_lock(tmp_path):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        assert result.returncode != 0 and b"Another pipeline writer" in result.stderr
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_resume_screening_matches_uninterrupted(tmp_path, config, monkeypatch):
    import pyarrow as pa
    import pipeline
    import io
    config["screen_batch_size"] = 2
    rows, values = [], []
    rng = np.random.default_rng(42)
    for i in range(7):
        im = Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        values.append({"bytes": buf.getvalue(), "path": None})
        rows.append({"image_id": str(i), "source_index": i, "screen_rank": str(i)})
    table = pa.table({"image": values})
    monkeypatch.setattr(pipeline, "history", lambda: HashIndex())
    root1, root2 = tmp_path / "resumed", tmp_path / "whole"
    monkeypatch.setattr(pipeline, "ROOT", root1)
    s = Store(root1 / "runtime/state.sqlite", {})
    assert pipeline.screen(s, rows, table, config, stop_after_batches=1) is False
    s.close()
    s = Store(root1 / "runtime/state.sqlite", {})
    # Simulate a damaged exported image after an otherwise committed batch.
    target = root1 / s.get("screen")["0"]["candidate_path"]
    target.write_bytes(b"truncated")
    assert pipeline.screen(s, rows, table, config) is True
    resumed = s.get("screen")
    assert sha(target) == resumed["0"]["image_sha256"]
    s.close()
    monkeypatch.setattr(pipeline, "ROOT", root2)
    whole = Store(root2 / "runtime/state.sqlite", {})
    assert pipeline.screen(whole, rows, table, config) is True
    assert whole.get("screen") == resumed
    whole.close()
