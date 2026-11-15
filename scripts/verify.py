"""Verify published artifacts before rebuilding them; never access the network."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = (ROOT / "PUBLICATION_SHA256SUMS.txt").read_text().splitlines()
    for line in lines:
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Unsafe checksum path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Artifact missing or changed: " + name)
    print(f"Verified {len(lines)} published artifacts")


if __name__ == "__main__":
    main()
