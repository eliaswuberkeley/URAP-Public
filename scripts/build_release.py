"""Build a deterministic, fully hash-checked FairFace image release."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "fairface-v1.0.0"
NAME = "fairface-matched-v1.0.0.zip"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def entry(name):
    info = zipfile.ZipInfo(name, date_time=(2026, 10, 6, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build(image_dir, destination):
    data = ROOT / "fairface/data"
    with (data / "selected_images.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4000 or len({r["image_id"] for r in rows}) != 4000:
        raise ValueError("Expected 4,000 distinct images")
    members = {}
    for row in rows:
        relative = Path(row["image_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe image path")
        source = image_dir / relative
        content = source.read_bytes()
        if digest(content) != row["image_sha256"]:
            raise ValueError("Image checksum mismatch: " + row["image_id"])
        members["private/dataset/" + relative.as_posix()] = content
    for name, source in {
        "release-info/selected_images.csv": data / "selected_images.csv",
        "release-info/matched_pairs.csv": data / "matched_pairs.csv",
        "release-info/DATA_LICENSE.md": ROOT / "fairface/DATA_LICENSE.md",
        "release-info/README.md": ROOT / "fairface/README.md",
    }.items():
        members[name] = source.read_bytes()
    members["release-info/IMAGE_SHA256SUMS.txt"] = "".join(
        f"{digest(content)}  {name}\n" for name, content in sorted(members.items())
        if name.startswith("private/dataset/")
    ).encode()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / NAME
    temporary = target.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(entry(name), content)
    temporary.replace(target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(members):
            raise ValueError("ZIP verification failed")
        for name, content in members.items():
            if archive.read(name) != content:
                raise ValueError("ZIP member drift: " + name)
    with target.open("rb") as handle:
        archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    record = {"tag": TAG, "asset": NAME, "sha256": archive_hash,
              "bytes": target.stat().st_size, "images": len(rows), "matched_pairs": 2000,
              "images_by_split": dict(sorted(Counter(r["split"] for r in rows).items())),
              "selected_images_sha256": digest((data / "selected_images.csv").read_bytes()),
              "matched_pairs_sha256": digest((data / "matched_pairs.csv").read_bytes()),
              "original_backgrounds": True, "human_expression_review": False}
    expected = ROOT / "release-manifest.json"
    if expected.exists() and json.loads(expected.read_text()) != record:
        raise ValueError("Release differs from frozen public release manifest")
    (destination / "release-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    (destination / "SHA256SUMS.txt").write_text(f"{archive_hash}  {NAME}\n")
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "fairface/private/dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    build(args.image_dir, args.output)


if __name__ == "__main__":
    main()
