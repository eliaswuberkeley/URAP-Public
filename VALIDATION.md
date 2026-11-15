# Publication validation

The public export was checked locally on November 15, 2026 using Python 3.11.
The historical research directories were not modified.

- Installed the published replay requirements in a new virtual environment.
- Reproduced all 2,000 matched pairs, identifiers and splits from frozen annotations.
- Independent metadata audit passed: 4,000 unique source indices, 1,000 four-image
  matching sets, no recorded historical selection overlap, correct group/split labels.
- Passed 19 curation/storage/comparison tests, including interruption recovery,
  matching calipers, cardinality, deterministic ordering and pixel-exact composites.
- Passed five IAT reproduction tests, preserving every word, count and valence boundary.
- Rebuilt all six curation figures and both IAT figures without inference or image
  access. All 600 saved responses were reparsed and validated against frozen hashes.
- Independently downloaded the pinned public Parquet source and reconstructed all
  4,000 PNGs. Every source-image and exported-image hash matched the selection manifest.
- Built the deterministic ZIP twice: once from the original local images and once
  from independently reconstructed images. Both produced the archive SHA-256 in
  `release-manifest.json`. ZIP integrity and every member were checked.
- Ruff's E4/E7/E9/F checks passed for the public Python code.
- Publication candidates were checked for provider-key/private-key patterns, inline
  image payloads, and signed URLs. Face images, model weights and account credentials
  are excluded from Git. The image ZIP is an explicitly approved public Release.

`PUBLICATION_SHA256SUMS.txt` identifies the final published code/data/plots; run
`python scripts/verify.py` on a fresh clone before regenerating outputs.
`requirements.txt` pins direct dependencies; `requirements.lock.txt` records the
complete clean replay environment. The tag-triggered GitHub workflow repeats tests,
source reconstruction and archive verification before publishing the download.

The first Linux x64 workflow passed all 19 tests and pair replay but failed the
PNG hash check during reconstruction. Version 1.0.1 uses a macOS ARM64 runner to
match the locally verified encoder platform. Package-version pins alone did not
ensure identical encoded images. The check was not relaxed and no mismatched
image files were published. On other platforms, use the prebuilt Release ZIP.

## Limits

This validates artifact integrity and computational replay, not the truth of
automated expression labels or the absence of model bias. Full classifier inference
and fresh Gemma training were not rerun. Plot bytes can change with fonts/renderers,
while numerical counts and selection must remain fixed. Full historical curation
setup needs additional source/model dependencies; the portable `reproduce.py` path
uses the included frozen annotations. No external model calls or RunPod were used.

The public `analyze.py` uses a verified image-byte-size manifest so image-free plotting
works. Public documentation adds download/licensing/navigation instructions. These
publication changes do not change pair membership, response data or scientific calculations.
