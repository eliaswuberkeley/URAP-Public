# October 6 methods and rationale

## Research objective

Construct a fresh Black-White matched FairFace sample for future grounded model
training and held-out evaluation. The desired future outcome is reduced excess
negativity associated with race when visible expression and other measured image
properties are comparable, without removing accurate negative descriptions or
collapsing response diversity. This curation run neither trains a model nor selects
images using an IAT outcome. It cannot establish that an adapter improves fairness.

## Sample size and splitting

The target is 2,000 two-image pairs, not 2,000 images. Each pair contains one image
with the publisher label Black and one with the publisher label White. Both images
have the same publisher gender and age band and the same classifier expression
stratum. Two gender-specific pairs form a four-image matching set with a shared
split. This maintains joint age/expression balance in all four race/gender cells
within train, development, and test. Target images per split: 2,400, 800, 800.

FairFace lacks identity identifiers here. Images are unique and approximately
deduplicated, but person-disjointness is not verified. Separate pictures of the same
person could evade the screening. Images showing different people are not causal
race-only counterfactuals. The 20-69 age scope is inherited from September 29.

## Source and exclusions

Use only the publisher train split, configuration 0.25, at revision
`54d573cdb8b5af490ba8da9da2799628f6e5c496`. Verify the two local Arrow-file hashes
and the source class-label mapping against frozen September 29 records. Research
train/dev/test allocations are not the original publisher split names.

Exclude recorded historical FairFace indices, the entire September 29 FairFace
selection, and matches to the combined historical plus September 29 image-hash
registry. Screening uses identical decoded RGB pixels or both perceptual-hash
distance <=4 and difference-hash distance <=6. Deterministic greedy representatives
avoid reuse of approximately duplicate pictures in the current release. The
historical registry is conservative, not proof of complete prior non-exposure.

## Technical screening and expressions

All metadata-eligible candidates are decoded, EXIF-oriented, and exported as RGB
PNGs without resizing. Reject unreadable or undersized (<64 pixels minimum side)
images. Compute grayscale brightness, contrast, adjacent-pixel edge energy,
clipping fractions, and image hashes. Do not exclude images for dark pixels.

Annotate nested, deterministic candidate waves (200, 400, 800, then all per
age/race/gender cell), stopping when matching support reaches the frozen target.
This is a computation-saving stopping rule based only on matching feasibility,
not language-model bias outcomes. Frozen candidate-wave reports document it.

Classifier: `trpakov/vit-face-expression`, revision
`ef0bc6fc34241b6587e7e009e7711357be28c024`. Reuse previous annotations only when
the model revision and exported-image SHA-256 agree. Otherwise run deterministic
local CPU inference, batch size 16 (pad the final batch), four Torch threads, seed
20261006. The implementation records version pins and rejects contract drift when
resuming. Frozen annotations, rather than rerunning a neural network on arbitrary
hardware, are the authoritative source for exact selection reproduction.

Happy maps to positive; neutral to neutral; sad, angry, fear, and disgust map to
negative. Confidence below 0.6 and surprise map to uncertain. Keep the detailed
top label in the matching stratum: a high-confidence sad face cannot be matched
to an angry face merely because both were called negative. Uncertain matches
remain uncertain and require a separate future sensitivity analysis.

## Matching and quality constraints

Within each gender/age/expression stratum, use SciPy's linear-sum assignment.
Disallow edges where expression confidence differs by >0.15 or where contrast or
log edge energy differ by >2 pooled candidate standard deviations. Assign a
penalty greater than every allowed edge combined to forbidden edges, maximizing
feasible pair count before minimizing the sum of squared standardized quality
differences plus squared scaled confidence difference. Calipers are pragmatic
design choices, not clinically validated thresholds.

Use the minimum capacity across Female and Male pairs in every age/expression
stratum. Allocate age quotas as evenly as common support permits; allocate
expression quotas proportionally to observed candidate support, capped by common
support. Within quotas prefer lower-cost pairs with deterministic ties. This is a
balanced, quality-matched research sample, not a representative population sample.
Confidence is an uncertainty proxy, not a calibrated expression-intensity measure.

Brightness is diagnostic-only because global brightness conflates lighting,
background, and skin reflectance. Equalizing it could systematically select
unrepresentative images. Quality proxies do not fully control pose, occlusion,
crop semantics, expression intensity, or camera differences. Report residual
differences and uncertain-expression prevalence, not a claim that all attributes
are equal. No independent human review or demographic inference is performed.

## Interruption and validation

SQLite transactions commit one completed screening/inference batch at a time;
FULL synchronization and fullfsync are enabled. File exports use atomic same-
directory replacement after flushing, then flush the directory. An OS file lock
prevents simultaneous writers and is released on process death. Checkpoints have
per-record SHA-256 digests and SQLite integrity checks. Restart verifies image
hashes, repairs exports from the pinned source if needed, and skips committed
annotations. Partial work is repeatable. Tests kill a child process during an
uncommitted transaction, leave partial output files, corrupt a cached image, and
compare restarted screening with uninterrupted screening. These checks cannot
guarantee protection from hardware/filesystem failure; retain external backups.

Only publish a ready marker after checking image hashes, uniqueness, historical
and within-release duplicate screening, pair linkage, matching calipers, and
split-level joint balance. Final artifact checksums and a ZIP integrity test make
transfer errors detectable. The original run treated face images as private research
inputs. Subsequent owner approval authorized a separately versioned public GitHub
Release with attribution and dataset terms; images remain outside Git history.

## Citation

Karkkainen, K., and Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation. IEEE/CVF Winter
Conference on Applications of Computer Vision (WACV).
https://github.com/joojs/fairface
