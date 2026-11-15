"""Losslessly arrange frozen FairFace pairs into resumable comparison PNGs."""
from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
import re
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from storage import atomic_bytes, read_csv, sha, write_csv, write_json, writer_lock  # noqa: E402

OUT = ROOT / "private/side_by_side_comparisons"
SPLITS = {"train": 1200, "development": 400, "test": 400}
POLICY = {"version": 1, "left_publisher_race": "Black", "right_publisher_race": "White",
          "source_size_pixels": [224, 224], "comparison_size_pixels": [448, 224],
          "mode": "RGB", "gap_pixels": 0, "resize": False, "crop": False,
          "text_overlay": False, "purpose": "visual_comparison_not_a_new_evaluation_protocol"}


def load_inputs():
    ready = json.loads((ROOT / "data/DATASET_READY.json").read_text())
    source_hashes = {}
    for name in ("data/matched_pairs.csv", "data/selected_images.csv"):
        source_hashes[name] = sha(ROOT / name)
        if source_hashes[name] != ready["files"][name]:
            raise ValueError("Frozen input changed: " + name)
    pairs = read_csv(ROOT / "data/matched_pairs.csv")
    image_rows = read_csv(ROOT / "data/selected_images.csv")
    images = {r["image_id"]: r for r in image_rows}
    if len(pairs) != 2000 or len(image_rows) != len(images) or len(images) != 4000:
        raise ValueError("Expected exactly 2,000 pairs and 4,000 unique source images")
    if Counter(p["split"] for p in pairs) != SPLITS:
        raise ValueError("Frozen split counts differ")
    used = []
    for pair in pairs:
        relative_path(pair)
        for side, race in (("black", "Black"), ("white", "White")):
            row = images[pair[side + "_image_id"]]
            if row["source_race"] != race or row["pair_id"] != pair["pair_id"]:
                raise ValueError("Image-to-pair linkage mismatch")
            for field in ("split", "source_gender", "age_band", "expression_class", "expression_label"):
                if row[field] != pair[field]:
                    raise ValueError("Pair metadata mismatch: " + field)
            used.append(row["image_id"])
    if len(set(p["pair_id"] for p in pairs)) != 2000 or len(set(used)) != 4000:
        raise ValueError("Duplicate pair or source image assignment")
    return pairs, images, source_hashes


def relative_path(pair):
    if pair["split"] not in SPLITS or not re.fullmatch(r"ffpair_[0-9a-f]{20}", pair["pair_id"]):
        raise ValueError("Unsafe or unexpected comparison filename")
    return Path(pair["split"]) / (pair["pair_id"] + ".png")


def source_image(row):
    base = ROOT / "private/dataset"
    path = (base / row["image_path"]).resolve()
    if not path.is_relative_to((base / "images").resolve()):
        raise ValueError("Source image path outside frozen image folder")
    if sha(path) != row["image_sha256"]:
        raise ValueError("Source image checksum mismatch: " + row["image_id"])
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (224, 224):
            raise ValueError("Expected an unchanged 224x224 RGB source image")
        return im.copy()


def exact_halves(path, left, right):
    try:
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != (448, 224):
                return False
            return (im.crop((0, 0, 224, 224)).tobytes() == left.tobytes()
                    and im.crop((224, 0, 448, 224)).tobytes() == right.tobytes())
    except (OSError, ValueError):
        return False


def write_comparison(path, left, right):
    if any(im.mode != "RGB" or im.size != (224, 224) for im in (left, right)):
        raise ValueError("Both sources must be unmodified 224x224 RGB images")
    if path.exists() and exact_halves(path, left, right):
        return False
    combined = Image.new("RGB", (448, 224))
    combined.paste(left, (0, 0))
    combined.paste(right, (224, 0))
    data = io.BytesIO()
    combined.save(data, format="PNG", optimize=False)
    atomic_bytes(path, data.getvalue())
    if not exact_halves(path, left, right):
        raise ValueError("Comparison pixels do not match originals")
    return True


def readme():
    return """# October 6 side-by-side FairFace comparisons

2,000 comparison PNGs contain the same 4,000 selected source images.
Every PNG is 448 x 224 pixels: publisher Black label on the LEFT,
publisher White label on the RIGHT. These are dataset annotations,
not labels inferred or independently verified for this export.

Folders: train/ (1,200 pairs), development/ (400), test/ (400).
The filename is the original pair_id. Original pixels are preserved
without scaling, cropping, color changes, gaps, or text overlays.

manifest.csv retains all matched_pairs.csv columns and adds:
- comparison_path: PNG path relative to this folder.
- comparison_sha256: SHA-256 of the combined PNG bytes.
- comparison_width / comparison_height: dimensions in pixels.
- left_image_sha256 / right_image_sha256: original source PNG hashes.
matched_pairs.csv is an exact copy of the original pair CSV.
The original column definitions are included in COLUMN_GLOSSARY.csv.
SHA256SUMS verifies this folder's deliverable files.
READY.json is written only after all 2,000 images pass validation.

These are matched photos of different people, not race-only causal
counterfactuals. Automated expression labels can be wrong. Pose,
occlusion and brightness are not fully controlled. Preserve the
train/development/test separation and retain the original single-image
inputs for existing benchmarks. Feeding these composites to a model
changes the input protocol; a fixed left/right order can also confound
evaluation. This export performs no training or benchmark inference.

Keep face-image files out of public GitHub uploads. Follow applicable
source-image rights and dataset terms, retaining attribution and the
modification description above. Source: https://github.com/joojs/fairface
Karkkainen and Joo, FairFace: Face Attribute Dataset for Balanced Race,
Gender, and Age for Bias Measurement and Mitigation, WACV 2021.
The FairFace repository lists CC BY 4.0.

Reproduce from the October 6 project root:
python scripts/build_comparisons.py
Verify without changing files:
python scripts/build_comparisons.py --verify-only
Restarting validates existing PNGs and repeats only missing/invalid
image writes. A completed image is never reused based on filename alone.
"""


def build():
    pairs, images, source_hashes = load_inputs()
    contract = {"policy": POLICY, "source_hashes": source_hashes}
    expected = {str(relative_path(p)) for p in pairs}
    actual = {str(p.relative_to(OUT)) for p in OUT.rglob("*.png")} if OUT.exists() else set()
    if actual - expected:
        raise ValueError("Unexpected PNGs in comparison folder; preserve and inspect rather than deleting")
    contract_path = OUT / "BUILD_CONTRACT.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Comparison source or layout contract changed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "READY.json").unlink(missing_ok=True)
    write_json(contract_path, contract)
    records = []
    written = 0
    start = time.monotonic()
    for i, p in enumerate(pairs, 1):
        left_row, right_row = images[p["black_image_id"]], images[p["white_image_id"]]
        left, right = source_image(left_row), source_image(right_row)
        relative = relative_path(p)
        target = OUT / relative
        written += write_comparison(target, left, right)
        records.append({**p, "comparison_path": str(relative), "comparison_sha256": sha(target),
                        "comparison_width": 448, "comparison_height": 224,
                        "left_image_sha256": left_row["image_sha256"],
                        "right_image_sha256": right_row["image_sha256"]})
        if i % 100 == 0:
            progress = {"completed": i, "total": len(pairs), "written": written,
                        "reused": i - written, "elapsed_seconds": round(time.monotonic() - start, 2)}
            write_json(ROOT / "runtime/side_by_side_progress.json", progress)
            print(json.dumps(progress), flush=True)
    write_csv(OUT / "manifest.csv", records)
    atomic_bytes(OUT / "matched_pairs.csv", (ROOT / "data/matched_pairs.csv").read_bytes())
    atomic_bytes(OUT / "COLUMN_GLOSSARY.csv", (ROOT / "docs/COLUMN_GLOSSARY.csv").read_bytes())
    atomic_bytes(OUT / "README.md", readme().encode())
    result = {"pairs": len(records), "source_images": len(images), "pairs_by_split": SPLITS,
              "pixel_preservation_verified": True, "source_image_hashes_verified": 4000,
              "source_files_modified": False, "comparison_dimensions": [448, 224],
              "comparison_image_bytes": sum((OUT / r["comparison_path"]).stat().st_size for r in records)}
    write_json(OUT / "validation.json", result)
    members = [OUT / r["comparison_path"] for r in records]
    members += [OUT / name for name in ("manifest.csv", "matched_pairs.csv", "COLUMN_GLOSSARY.csv",
                                        "README.md", "BUILD_CONTRACT.json", "validation.json")]
    atomic_bytes(OUT / "SHA256SUMS", ("\n".join(sha(p) + "  " + str(p.relative_to(OUT))
                                                 for p in sorted(members)) + "\n").encode())
    write_json(OUT / "READY.json", {"status": "complete", "pairs": 2000,
                                   "checksums_sha256": sha(OUT / "SHA256SUMS"),
                                   "source_hashes": source_hashes})
    print(json.dumps({**result, "written": written, "reused": len(records) - written,
                      "folder": str(OUT)}), flush=True)


def verify():
    pairs, images, source_hashes = load_inputs()
    marker = json.loads((OUT / "READY.json").read_text())
    if marker["status"] != "complete" or marker["source_hashes"] != source_hashes:
        raise ValueError("Incomplete or changed comparison release")
    if sha(OUT / "SHA256SUMS") != marker["checksums_sha256"]:
        raise ValueError("Checksum manifest changed")
    checks = 0
    for line in (OUT / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        target = (OUT / name).resolve()
        if not target.is_relative_to(OUT.resolve()) or sha(target) != checksum:
            raise ValueError("Comparison checksum failure: " + name)
        checks += 1
    rows = read_csv(OUT / "manifest.csv")
    if len(rows) != len(pairs):
        raise ValueError("Comparison manifest row count mismatch")
    if {p.relative_to(OUT).as_posix() for p in OUT.rglob("*.png")} != {str(relative_path(p)) for p in pairs}:
        raise ValueError("Missing or unexpected comparison PNGs")
    for p, r in zip(pairs, rows, strict=True):
        if any(r[k] != v for k, v in p.items()):
            raise ValueError("Original pair metadata changed in comparison manifest")
        path = OUT / relative_path(p)
        if r["comparison_path"] != str(relative_path(p)) or sha(path) != r["comparison_sha256"]:
            raise ValueError("Incorrect comparison linkage")
        left, right = images[p["black_image_id"]], images[p["white_image_id"]]
        if r["left_image_sha256"] != left["image_sha256"] or r["right_image_sha256"] != right["image_sha256"]:
            raise ValueError("Incorrect source hashes")
        if not exact_halves(path, source_image(left), source_image(right)):
            raise ValueError("Comparison pixels changed")
    if sha(OUT / "matched_pairs.csv") != source_hashes["data/matched_pairs.csv"]:
        raise ValueError("Copied pair CSV differs from source")
    print(json.dumps({"passed": True, "comparison_images_verified": len(rows),
                      "source_images_verified": len(images), "artifact_checksums_verified": checks,
                      "pairs_by_split": dict(Counter(r["split"] for r in rows))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    with writer_lock(ROOT):
        verify() if args.verify_only else build()
