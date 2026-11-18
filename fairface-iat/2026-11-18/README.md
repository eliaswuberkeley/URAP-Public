# Does Gemma's word choice differ for expression-matched FairFace portraits?

**November 18, 2026 | Base Gemma 3 12B IT, no adapter.** On this panel, Gemma chose no negative words for positive or neutral expressions. For negative expressions, it chose negative words more often for Black portraits (12.50%) than White portraits (1.67%). The difference was concentrated in a few portraits and was not statistically significant.

| Human-reviewed expression (full-size PNG) | Black: negative words | White: negative words | Black minus White | Paired 95% interval | Exact p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| [Positive](plots/iat61_positive.png) | 0/360 (0%) | 0/360 (0%) | 0 pp | [0, 0] pp | 1.000 | 1.000 |
| [Neutral](plots/iat61_neutral.png) | 0/360 (0%) | 0/360 (0%) | 0 pp | [0, 0] pp | 1.000 | 1.000 |
| [Negative](plots/iat61_negative.png) | 45/360 (12.50%) | 6/360 (1.67%) | +10.83 pp | [-0.83, +25.00] pp | 0.219 | 0.656 |

Each denominator is 24 portraits per race x 5 responses x 3 words. **The statistical unit is a matched pair, not a word or a repeated answer.** In the negative-expression panel, Black portraits had more negative words in 4 pairs, White portraits in 2, and 18 tied. Three Black portraits accounted for 42 of the 45 negative selections. The zero-width intervals in the other panels reflect zero observed negative words, not proof of zero bias elsewhere.

## Three word-count charts

The vertical axis counts times a word was selected. All panels share the same vertical scale. Each chart shows all 61 words, with positive words on the left and negative words on the right. Within each half, words are sorted by pooled frequency for that panel.

![Positive-expression pairs: zero negative words selected for either group.](plots/iat61_positive.png)

![Neutral-expression pairs: zero negative words selected for either group.](plots/iat61_neutral.png)

![Negative-expression pairs: 45 negative word selections for Black portraits and 6 for White portraits.](plots/iat61_negative.png)

## Sample and method

- From the [October matched FairFace collection](../../fairface/README.md), a single reviewer approved original test-split pairs as having matching visible expressions. Black and White are the source dataset labels. Each selected pair also matches on age band and gender. These are ratings of visible expression, not claims about a person's emotional state.
- Before inference, the review app showed no recorded Gemma reveal for these approvals. Neither member had appeared in the indexed earlier base-IAT panels. A seeded draw selected 24/41 approved positive pairs, 24/24 neutral and 24/25 negative without replacement (ascending SHA-256 of `2026111803:expression:pair_id`). This panel is **not** the [earlier 120-portrait development benchmark](../README.md) and is not representative of all FairFace.
- Each of the 144 original portraits received the same [61-word prompt](../iat_protocol.py); the model did not receive the expression or race label. Revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`, H100, bfloat16, eager attention, Transformers 4.50.0. Five distinct reproducible seeds per portrait; sampled decoding with temperature 1.0, top-p 1.0 and up to 30 new tokens. Of 720 sampled answers, 13 needed a greedy formatting retry (up to 20 new tokens). All 720 final answers yielded three words under the historical parser.
- Intervals use 100,000 pair-bootstrap draws. Two-sided exact sign-flip tests compare Black versus White within pairs; Holm adjustment covers the three primary race-gap tests. They do not test differences *between* expressions. The negative-panel gap was +10.21 pp when restricted to original samples that exactly followed the requested format.

## Scoring labels

The `analysis` column in [pair_rates.csv](results/pair_rates.csv) identifies the scoring rule:

- **`primary`** (used for the charts): Counts the first three words the parser recognizes from the 61-word list, even if the answer includes extra text. If it finds fewer than three, it scores one formatting retry instead. All 720 responses were scored under this rule, including 13 retries.
- **`sampled_strict`** (format check): Counts only original answers that contain exactly three comma-separated words from the list and no extra text. Capitalization and spacing around commas do not matter. Retries are excluded; 628 of 720 original answers qualified.

## Data and local reproduction

| File | Contents |
|---|---|
| [Raw responses (CSV)](data/responses.csv) | One row per portrait and repetition; sampled text, fallback text, parsed words, seed, validity flags and source pair/image IDs. The image path is a historical run-relative identifier, not a public image URL. |
| [Word counts (CSV)](results/word_counts.csv) | Counts behind every bar, by expression and source race, including words never selected. |
| [Pair-level results (CSV)](results/pair_rates.csv) | Within-pair negative counts, denominators, rates and differences for both scoring rules. |
| [Full statistics (JSON)](results/statistics.json) | Primary and strict-format results, validity, intervals, p-values and retry counts. |
| [Reproduction script](reproduce.py) | Checks all 720 completions against the public prompt parser, source manifest and seeds; rebuilds all three plots and verifies the published statistics. |

From the root of a clone, run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r fairface-iat/requirements.txt
.venv/bin/python fairface-iat/2026-11-18/reproduce.py
```

Rebuilt PNGs go to `fairface-iat/2026-11-18/runtime/rebuilt/`. This replays the analysis from saved responses; it does not rerun model inference, use cloud credits, or require the images. For the original portraits see the [existing dataset release](https://github.com/eliaswuberkeley/URAP-Public/releases/tag/fairface-v1.0.1) and [FairFace attribution and terms](../../fairface/DATA_LICENSE.md).
