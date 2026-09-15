# Synthetic breath VOC profiles and reproducible respiratory disease modelling

A resource combining **6,000 synthetic profiles**, **78 shared volatile organic
compounds**, a trained conditional **WGAN-GP**, and a patient-level evaluation benchmark.
The three classes are **asthma, bronchiectasis and COPD** (2,000 generated profiles each).
The source cohort contains 121 patients.

## Use the current release

- [Documentation and runnable examples](zenodo_release/README.md)
- [Data and compound dictionary](zenodo_release/data/)
- [Portable training and evaluation code](zenodo_release/code/)
- [Audited benchmark results](zenodo_release/BENCHMARK_SUMMARY.md)
- [Fold-specific samples and provenance](zenodo_release/results/audited/)
- [Trained models and preprocessing](zenodo_release/model/)

Run commands from the release folder:

```bash
cd zenodo_release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/validate_release.py
python code/benchmark_example.py
```

The release includes real-only, bootstrap, synthetic-only and augmentation
comparisons on fixed real test folds. It supports methodological reuse and
controlled sampling; generated profiles are not new patient measurements.

## Citation and provenance

Reserved Zenodo DOI: **10.5281/zenodo.21908977** (private draft; publication pending).
See [citation metadata](zenodo_release/CITATION.cff). Data are CC BY 4.0 and code is MIT.

Source: Kuo et al., [A clinical breathomics dataset](https://doi.org/10.1038/s41597-024-03052-2),
and [the version 6 source files](https://doi.org/10.6084/m9.figshare.23522490.v6).

Earlier repository files are preserved in [legacy/](legacy/). The correct condition
is bronchiectasis and the released generator is a conditional WGAN-GP.
CTGAN is a separate baseline in the benchmark.
