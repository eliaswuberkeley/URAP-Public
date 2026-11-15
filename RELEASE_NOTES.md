# FairFace matched sample v1.0.0

Frozen October 6, 2026 selection: 2,000 Black-White matched pairs, 4,000 original
224x224 portraits. Research splits: 2,400 training, 800 development, 800 test images.

Download **fairface-matched-v1.0.0.zip** under Assets for the photographs. GitHub's
automatic "Source code" archives contain code and metadata, not face images.
Unzip the image ZIP in the repository's `fairface/` directory. Verify the supplied
SHA256SUMS.txt before use. The archive includes per-image checksums and attribution.

Original backgrounds are retained. Expression labels are automated and may be wrong;
this is not a human-verified neutral dataset. Matching does not establish that race
is the only difference between photos. No individual identity-disjointness guarantee.

Images and derived selection tables: CC BY 4.0 with FairFace attribution; original
code: MIT. See fairface/DATA_LICENSE.md. No UTKFace images or model weights included.

The matching code, frozen inputs and 120-portrait IAT results are in this release's
source tree. This release does not represent a new model evaluation or new training.
Future dataset changes will use a new version/tag, preserving this release.
