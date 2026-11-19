# URAP Public

Public research code and reproducibility resources maintained by Elias Wu.

## Start Here

| Resource | Contents |
| --- | --- |
| [Matched FairFace dataset](fairface/README.md) | 2,000 pairs / 4,000 portraits; exact reconstruction, methods and column glossary. |
| [Public image download](https://github.com/eliaswuberkeley/URAP-Public/releases/tag/fairface-v1.0.1) | Versioned ZIP attachment, checksums and attribution. Images are not stored in Git history. |
| [Matched FairFace IAT results](fairface-iat/README.md) | Local base Gemma, 120 development portraits, 600 responses, full 61-word plot and raw data. |
| [Main IAT chart](fairface-iat/plots/fairface_base_61_word_counts_sorted.png) | Positive words left, negative words right, each half sorted by pooled frequency. |
| [Human-reviewed FairFace IAT: positive, neutral and negative expressions](fairface-iat/2026-11-18/README.md) | November 18: 24 test pairs per expression, 720 base-Gemma responses, three charts and paired statistical tests. |
| [Negative-expression IAT follow-up: 36 human-matched pairs](fairface-iat/2026-11-18/negative-36/README.md) | Pooled base-Gemma results, 61-word chart, paired significance test and all 360 responses. |
| [Publication validation](VALIDATION.md) | Checks performed, reproducibility limits and source preservation. |

The dataset has original backgrounds and automated expression labels. It is not a
background-normalized or human-verified neutral-expression set. The earlier development IAT uses
120 of the 4,000 images, including negative, neutral and positive expression estimates;
it is not a full-dataset evaluation or evidence that the model is unbiased. The November 18
study separately evaluates 144 human-reviewed test portraits from the collection.

Clone this repository and follow the matching or IAT README. Existing plots can be
reproduced locally without model API calls or a GPU. New inference and training are
separate tasks. Model weights and unrelated private research are not included.

## Versioning

`fairface-v1.0.1` preserves the October 6 selection. Changes to membership, labels,
splits or preprocessing should receive a new version and release, never silently
replace this dataset. Each release pins a source-code commit and archive checksum.
The release workflow reconstructs and verifies every image before publishing.
Version 1.0.1 fixes the build platform for exact PNG bytes; the earlier 1.0.0
source tag remains in history, but its Linux build stopped before publishing images.

## Licensing

- Original code and its accompanying original documentation published in this
  repository are licensed under the [MIT License](LICENSE), unless a file is
  explicitly marked otherwise.
- The MIT license does **not** relicense third-party code, datasets, images,
  annotations, or model weights. Their applicable licenses and attribution
  requirements remain in force.
- [FairFace](https://github.com/joojs/fairface) is published under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Any FairFace-derived
  dataset release must retain the applicable attribution and license information
  and document its curation or other changes. The MIT license is not a replacement
  for these dataset terms.

## FairFace Reference

Karkkainen, K., and Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation.* Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision, 1548-1558.

[Original paper](https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html)
