"""Reproduce the frozen selection and optionally reconstruct selected PNGs."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from matching import match  # noqa: E402
from storage import atomic_bytes, digest, read_csv, sha  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rebuild-images", action="store_true")
    p.add_argument("--source-dir", type=Path, help="Directory containing the two pinned training Arrow or Parquet shards")
    p.add_argument("--download", action="store_true", help="Explicitly download pinned FairFace Parquet shards into private/source")
    args = p.parse_args()
    ready = json.loads((ROOT / "data/DATASET_READY.json").read_text())
    validation = json.loads((ROOT / "data/validation.json").read_text())
    if sha(ROOT / "data/validation.json") != ready["validation_sha256"] or not validation["passed"]:
        raise ValueError("Validation record changed or did not pass")
    if sha(ROOT / "config/curation.json") != validation["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    for path, expected in ready["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Frozen artifact drift: " + path)
    config = json.loads((ROOT / "config/curation.json").read_text())
    pool = read_csv(ROOT / "data/matching_candidate_pool.csv")
    for r in pool:
        r["expression_confidence"] = float(r["expression_confidence"])
    pairs, _, report = match(pool, config)
    existing = read_csv(ROOT / "data/matched_pairs.csv")
    # CSV float round-tripping is compared numerically; identifiers/splits exactly.
    assert len(pairs) == len(existing)
    for actual, expected in zip(pairs, existing, strict=True):
        for k, value in actual.items():
            if isinstance(value, float):
                assert abs(value - float(expected[k])) < 1e-12, (k, value, expected[k])
            else:
                assert str(value) == expected[k], (k, value, expected[k])
    print(json.dumps({"selection_reproduced": True, "selected_pairs": report["selected_pairs"]}))
    if not args.rebuild_images:
        return
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    source = args.source_dir
    if args.download:
        from huggingface_hub import hf_hub_download
        source = ROOT / "private/source"
        spec = json.loads((ROOT / "data/frozen/source_files.json").read_text())
        filenames = [s.split(config["fairface_revision"] + "/", 1)[1]
                     for s in spec["fairface_dataset_info"]["download_checksums"] if "/train-" in s]
        paths = [Path(hf_hub_download(repo_id="HuggingFaceM4/FairFace", repo_type="dataset",
                                    revision=config["fairface_revision"], filename=name,
                                    local_dir=source, token=False)) for name in sorted(filenames)]
    else:
        if source is None:
            local = ROOT / "runtime/local_paths.json"
            if not local.exists():
                raise ValueError("Supply --source-dir or explicitly authorize --download")
            source = Path(json.loads(local.read_text())["source_dir"])
        paths = sorted(source.glob("fair_face-train-*.arrow")) or sorted(source.rglob("train-*.parquet"))
    if len(paths) != 2:
        raise ValueError("Expected exactly two ordered training shards")
    tables = []
    for path in paths:
        if path.suffix == ".arrow":
            with pa.memory_map(str(path), "r") as f:
                tables.append(ipc.open_stream(f).read_all())
        else:
            tables.append(pq.read_table(path))
    table = pa.concat_tables(tables)
    if len(table) != 86744:
        raise ValueError("Source row count mismatch")
    rows = read_csv(ROOT / "data/selected_images.csv")
    for row in rows:
        dest = ROOT / "private/dataset" / row["image_path"]
        if dest.is_file() and sha(dest) == row["image_sha256"]:
            continue
        raw = table["image"][int(row["source_index"])].as_py()["bytes"]
        if digest(raw) != row["source_image_sha256"]:
            raise ValueError("Selected source image hash mismatch")
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        data = io.BytesIO()
        rgb.save(data, format="PNG", optimize=False)
        if digest(data.getvalue()) != row["image_sha256"]:
            from PIL import features
            pixel_sha = digest(f"{rgb.width}x{rgb.height}|RGB|".encode() + rgb.tobytes())
            raise ValueError(
                "Reconstructed PNG hash mismatch for " + row["image_id"]
                + "; decoded_pixels_match=" + str(pixel_sha == row["pixel_sha256"])
                + "; zlib=" + str(features.version("zlib"))
                + ". Use the published Release ZIP for exact bytes, or the validated macOS ARM64 encoder."
            )
        atomic_bytes(dest, data.getvalue())
    print(json.dumps({"images_verified_or_rebuilt": len(rows)}))


if __name__ == "__main__":
    main()
