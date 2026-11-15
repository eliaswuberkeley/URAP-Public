# Matched FairFace: 2,000 pairs / 4,000 portraits

This is the frozen October 6, 2026 selection, published for reproducibility in
November. Each pair contains one Black-labeled and one White-labeled original
portrait. Labels are the dataset publisher's categories, not verified identities.

## Download

[Versioned image download](https://github.com/eliaswuberkeley/URAP-Public/releases/tag/fairface-v1.0.0)
contains `fairface-matched-v1.0.0.zip`, its checksums, and a release manifest.
The ZIP contains 4,000 individual PNGs, not 2,000 side-by-side composites.
The files have **original backgrounds**. Brightness normalization failed its
research quality checks and is not part of this release.

Unzip into this directory to create `private/dataset/`. No face images are stored
in Git history. `scripts/build_comparisons.py` can create side-by-side pairs later.

## What is matched?

- 1,000 images in each Black/Female, White/Female, Black/Male, White/Male cell.
- 1,200 training pairs, 400 development pairs, 400 test pairs.
- Same publisher age band (20-29 through 60-69), gender and estimated detailed
  expression within each pair; balanced expression/age counts across groups.
- Expression-confidence and contrast/detail calipers; deterministic minimum-cost
  matching within supported strata. Brightness is measured, not forced equal.
- Automated expression estimates use `trpakov/vit-face-expression` at revision
  `ef0bc6fc34241b6587e7e009e7711357be28c024`. They can be wrong or biased;
  464/4,000 images have an uncertain coarse expression class. This is not a
  human-reviewed neutral-expression dataset.
- No bias-test responses influenced sample selection. No verified person-disjoint
  guarantee: image hashes and source-index exclusions do not establish identity.

See [methods](docs/METHODS.md), [column glossary](docs/COLUMN_GLOSSARY.csv),
[configuration](config/curation.json), [selected portraits](data/selected_images.csv),
[pair table](data/matched_pairs.csv), and [results](RESULTS.md).
`pair_id` means two matched photographs, not a DPO chosen/rejected response pair.

## Reproduce without images or a GPU

From the repository root, on Linux or macOS with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify.py
.venv/bin/python fairface/scripts/reproduce.py
.venv/bin/python fairface/scripts/audit_release.py
.venv/bin/python fairface/scripts/analyze.py
.venv/bin/python -m pytest fairface/tests -q
.venv/bin/python fairface-iat/build.py
```

Selection replay uses the frozen annotation/candidate CSVs and reproduces every
pair ID and split. Plotting uses those same tables; `image_file_sizes.csv` replaces
a publication-only dependency on local images. Scientific calculations are unchanged.
Exact plot bytes can depend on platform and fonts; numeric data must agree.
Run the checksum verification before rebuilding outputs, since regenerated PDFs
may include new renderer metadata.

## Reconstruct the exact image files instead of downloading the ZIP

```sh
.venv/bin/python fairface/scripts/reproduce.py --rebuild-images --download
.venv/bin/python scripts/build_release.py
```

This explicitly downloads about 500 MB of public FairFace training Parquet shards,
then reconstructs and SHA-256-checks all 4,000 selected PNGs. It requires roughly
1 GB of free disk plus working memory. It does not run an API or expression model.
Source is `HuggingFaceM4/FairFace`, config `0.25`, publisher `train`, pinned to
`54d573cdb8b5af490ba8da9da2799628f6e5c496`. This publisher split differs from our
research train/development/test allocation. Cached shards can be supplied using
`--source-dir PATH` instead of `--download`.

## Full curation versus exact replay

`src/pipeline.py` preserves the original screening, duplicate checks, FER annotation
and resumable SQLite implementation for inspection. Its original `freeze` stage
expects a historical research workspace and model cache; it is **not** the portable
entry point for a fresh clone. Use `scripts/reproduce.py` for exact sample recovery.
Rerunning the classifier from scratch additionally needs its pinned model weights,
Torch/Transformers, the referenced historical exclusions and environment setup.
Those model weights are not bundled. Frozen annotations and provenance are included
so recovering this selection does not depend on re-inference or API availability.

Some frozen historical registries include September 29 UTKFace metadata because
cross-study duplicate screening used them. No UTKFace images are distributed here.
Historical workspace paths in provenance describe origins, not paths to create.

## Terms and citation

See [DATA_LICENSE.md](DATA_LICENSE.md). MIT applies only to original code, not
FairFace photographs, publisher annotations or third-party model weights.

Karkkainen, K. and Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation*. Proceedings of WACV.
[Original dataset](https://github.com/joojs/fairface).
