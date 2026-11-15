# Matched FairFace IAT results for sharing

Prepared November 14, 2026 from frozen October 6 results. This is a new layout,
not a new model evaluation. No model calls, training, RunPod session, or image edits
were performed. Historical source files are unchanged. This publication copy adds
public dataset/code links; it does not change the frozen responses or protocol.

## Start here

- `plots/fairface_base_61_word_counts_sorted.png`: the main chart, also available as PDF.
- `plots/fairface_dataset_and_iat_sample.png`: dataset splits and evaluation composition.
- `RESEARCHER_NOTE.txt`: short explanation ready to send to a colleague.
- `results/sorted_word_counts.csv`: all 61 words in exactly the chart order.
- `data/base_responses.csv`: 600 original response records, including unedited model text.
- `data/evaluated_images.csv`: the 120 images actually tested, with pair IDs and hashes.
- `data/curated_images.csv` and `data/matched_pairs.csv`: frozen full-dataset metadata
  for all 4,000 individual images / 2,000 Black-White matches. No image files are included.

## What was evaluated?

The focal model is the local, untuned `google/gemma-3-12b-it`, pinned to
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. These are not hosted OpenRouter Gemma
responses and do not use a LoRA adapter. This package does not compare adapters.

The full October 6 curated sample has 2,000 Black-White image pairs: 1,200 training,
400 development, and 400 test pairs. It has 1,000 images in each source race/gender
cell: Black/Female, Black/Male, White/Female, White/Male. Age bands span 20-69 but
are not equally sized. Publisher labels are not verified self-identifications.

This IAT run used **60 development pairs / 120 individual portraits**, with five
responses per portrait. It did not use all 4,000 images, the training portraits,
or the reserved test images. The 60 pairs include 22 negative-expression, 18 neutral,
and 20 positive-expression pairs, using automated expression estimates. Responses
see one original portrait, not a side-by-side composite. Background brightness was
not normalized. This is separate from the original 12-face benchmark.

The matching protocol controlled source gender, age band, estimated expression class
and detailed expression label, with confidence and image-quality calipers. Different
people remain different in unmeasured ways; these are not causal race-only counterfactuals.
No independent human review or verified person-disjointness is claimed.

## Read the chart

The model is asked to choose three terms from a fixed 61-word vocabulary: 30 positive
and 31 negative. Each bar counts selections of that word across the 300 answers for
one race, totaling 900 word selections per race. All vocabulary entries remain,
including zeros. Positive words are left and negative words right. Within each half,
words are sorted by **Black + White pooled frequency**, descending; ties retain the
original vocabulary order. Each race's bars need not individually descend. Both
halves use the same y-axis; the small negative bars are not magnified.

The chart reproduces the historical first-three-recognized-words parser, including
the recorded greedy fallback. All 600 final responses provide three parsed words,
but only 536/600 raw sampled responses obey the exact format. Seven responses needed
a fallback, yielding 607 original generation calls. After fallback, 543/600 final
texts obey the exact format. Parsed validity is not the same as strict validity.
`iat_protocol.py` preserves the exact vocabulary, prompt and parser. The initial
generation used `max_new_tokens=30`, sampling, temperature 1.0, top_p 1.0; unspecified
top_k inherits 64 from the stored model configuration. A short invalid output triggers
greedy fallback with `max_new_tokens=20`. The saved per-response seed is retained.
See `reference/evaluation_manifest.json` for the original environment and model
defaults; its generation configuration is not a substitute for these call overrides.

## Interpretation

The pooled counts contain 39/900 negative selections for Black images (4.33%) and
36/900 for White images (4.00%), a +0.33 percentage-point gap. The earlier report
instead equally weighted the three expression classes: 4.01% versus 3.97%, a +0.03
percentage-point gap. Both calculations use the same frozen responses, and the
replot verifies the standardized rates against the historical results.

Small aggregate gaps do not establish absence of bias. In the negative-expression
stratum, the Black-minus-White gap is +4.55 percentage points; in the neutral stratum
it is -4.44; in the positive stratum it is zero. These descriptive subgroup patterns
can cancel. Negative words can also be appropriate for visibly negative expressions.
Repeated answers from one image are not independent people. This development panel
has been inspected repeatedly and is exploratory, not a confirmatory test. No new
p-values or significance claims are introduced by this replot.

## Reproduce the figures, without images or a GPU

Use Python 3.11 and run from this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python build.py
.venv/bin/python -m unittest -v
shasum -a 256 -c SHA256SUMS.txt
```

Direct plotting dependencies are pinned. Fonts/renderer/platform can affect exported
image bytes, but the plotted counts, order and sample membership must match exactly.
`build.py` validates source checksums, every evaluated ID/hash/pair/split, all 600
parsed answers, and agreement with the historical standardized rates before plotting.
`reference/provenance.json` records original workspace-relative source paths and hashes.
The `--freeze-from` option was used once to assemble this package; recipients should
not use it. This package reproduces plots from recorded responses, not fresh Gemma
inference or complete dataset construction.

## Dataset access and matching code

The [public Release](https://github.com/eliaswuberkeley/URAP-Public/releases/tag/fairface-v1.0.1)
contains all 4,000 original individual images, including these 120 evaluated portraits.
Use `data/evaluated_images.csv` to locate them; image paths are relative to the
release's `private/dataset/`. Verify hashes against `data/curated_images.csv`.

The [matching/reconstruction package](../fairface/README.md) in this same repository
includes `scripts/reproduce.py`, configuration, frozen candidate/annotation CSVs,
source-download instructions and dataset terms. No private-repository access is needed.
The owner's Drive exports named **FairFace 4000 unpaired** and **FairFace 2000 paired**
are the individual and side-by-side presentations of this selection; those Drive
copies themselves have not been byte-verified here.

Dataset citation: Karkkainen and Joo, *FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation*, WACV 2021.
Original project reference: https://github.com/joojs/fairface . Follow source terms
and your group's sharing requirements. This package does not publish face images.

## Column definitions

`reference/dataset_columns.csv` explains every column of the full-image and pair
tables. `data/evaluated_images.csv` uses a subset of those same fields. `pair_id`
identifies two matched photos, not a chosen/rejected DPO response pair.

For `data/base_responses.csv`, each row is one image/repetition/model answer:

| Column | Meaning |
| --- | --- |
| image_id, image_sha256 | Stable image key and exported original PNG SHA-256. |
| image_path | Historical path relative to the October 6 training/evaluation project; not a bundled image. |
| match_id | Same identifier as `pair_id` in the curated dataset. |
| role | Evaluation allocation, `development_screen`. |
| source_split | Curated research split, `development`; unlike the publisher split in `curated_images.csv`. |
| race, gender, age_band | Publisher race/gender categories and age band. |
| expression_class, expression_label, expression_confidence | Automated coarse class, top label and uncalibrated classifier confidence. |
| sampled_raw | Original sampled output, before any fallback. |
| fallback_raw | Greedy replacement text when the sampled output had fewer than three recognized words; otherwise empty. |
| words | Final parsed vocabulary selections separated by `|`. |
| valid_three, strict_valid | Three parsed words; exact final-text format, respectively. |
| sampled_valid, sampled_strict_valid | Corresponding checks on sampled text alone. |
| negative_count, sampled_negative_count | Number of negative vocabulary selections in final/sample parsing. |
| generation_calls | Original calls for this record: one, or two including fallback. |
| seed, iteration | Recorded generation seed and repetition index 1-5. |
| model_key | `base`, no adapter. |
| created_at, latency_seconds | Original record timestamp and measured generation duration. |

`results/sorted_word_counts.csv`: `plot_position` is 1-61; `valence` is vocabulary
membership, not a judge rating; `within_valence_rank` is the position inside its half;
`word` is the vocabulary entry; `black_count` and `white_count` are selection counts;
`pooled_count` is their sum and the sorting key.

`results/negative_rates.csv`: `expression_group` is `all_pooled` or an automated
expression class; `race` is the source category; `images` and `responses` give sample
sizes; `selected_words` is the denominator; `negative_words` is the numerator;
`negative_rate_percent` is 100 times their ratio. `all_pooled` does not apply the
equal-expression weighting used in the historical headline metric.
