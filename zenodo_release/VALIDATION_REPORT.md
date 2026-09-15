# Verification report

Verified 15 September 2026 using the recorded Python 3.12 environment.

## Data and source

- All 6,000 generated profiles have 78 finite, non-negative features and the expected 2,000/2,000/2,000 labels.
- The combined synthetic CSV equals the concatenation of the three class files.
- The 121-profile source matrix rebuilds from the four fixed-version source files at relative and absolute tolerance 1e-12. Downloaded files match the source's checksums.
- The saved generator reproduces the archived sample within absolute tolerance 1e-5 in scaled model space. See generation_report.json for the measured maximum error.

## Benchmark

- All 45 neural fits completed: 15 each for WGAN-GP, CTGAN and TVAE.
- Five deterministic copula runs and 15 class-wise bootstrap samples complete the 65-run benchmark.
- Each sample has exactly 500 profiles per class, recorded feature order and disjoint training/test indices.
- Saved WGAN transforms match only their own real training partition. Float32 sample range checks allow 1e-6 scaled rounding tolerance.
- Accuracy, macro-F1 and macro-AUC for all 540 evaluation rows were recomputed from 396,776 archived predictions. Test-row coverage, true labels and probability normalization were also checked.
- Redundant copula seed outputs were verified array-identical before consolidation; the audit is in results/audited/consolidation.json.

## Fresh extraction

The ZIP passed its CRC check and was extracted into a new directory. Its manifest,
source rebuild, saved-model generation and standalone reuse example passed from
outside the original workspace. All 50 example rows reproduce the archived
metrics at relative and absolute tolerance 1e-10.

Pinned dependencies passed pip's fresh-resolution dry run. Execution used the
tested environment; this is not a claim of a second full dependency installation
or validation on every operating system. Checksums identify the archived bytes;
model retraining may vary across numerical environments.

The package is prepared for the author's private Zenodo draft, reserved DOI
10.5281/zenodo.21908977. Verification does not establish public deposition.
