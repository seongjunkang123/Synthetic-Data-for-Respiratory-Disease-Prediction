# Synthetic breath VOC profiles and a reproducible modelling benchmark

This resource combines **6,000 generated profiles across 78 chemically identified variables**
with a trained conditional WGAN-GP and a patient-level evaluation benchmark.
It supports repeatable comparisons of tabular synthesis methods, controlled sampling,
analysis-pipeline development and teaching.

The three classes are **asthma, bronchiectasis and COPD**, with 2,000 generated records each.
The model was fitted to 121 real patients (53 asthma, 35 bronchiectasis, 33 COPD).
Synthetic rows are model draws, not additional patient measurements.
The included source matrix is explicitly identified as real.

## Start here

Use Python 3.12 in a virtual environment. Commands run from this release directory;
scripts resolve their data paths relative to their own location.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python code/validate_release.py
python code/benchmark_example.py
~~~

On macOS, XGBoost requires the OpenMP runtime: **brew install libomp**.

For CSV inspection and a bootstrap-only example, use the smaller installation:

~~~bash
python -m pip install -r requirements-core.txt
python code/benchmark_example.py --bootstrap-only --output-dir /tmp/breathnet-bootstrap
~~~

The full example compares logistic regression trained on real records, class-balanced
bootstrap samples, fold-specific synthetic profiles, and real plus synthetic profiles.
Every condition uses its real training partition and the same untouched real test fold.
Outputs are archived in results/example/. See **BENCHMARK_SUMMARY.md** for audited results.

## Files

| Path | Contents |
|---|---|
| data/synthetic_asthma.csv | 2,000 generated asthma profiles |
| data/synthetic_bronchiectasis.csv | 2,000 generated bronchiectasis profiles |
| data/synthetic_copd.csv | 2,000 generated COPD profiles |
| data/synthetic_release_6000.csv | Exact concatenation of the three class files |
| data/voc_dictionary.csv | Two columns: pubchem_cid, iupac_name |
| data/real_121x78.csv | **Real patient-derived** harmonised source matrix |
| model/generator.keras, model/critic.keras | Archived full-cohort models |
| model/preprocessing.json | Saved log1p/min-max parameters |
| model/training_history.json | Archived training diagnostics |
| results/folds.json | Zero-based real training and test row indices |
| results/audited/synth/ | Fold/seed-specific samples and provenance sidecars |
| results/audited/cv_results2.csv | Per-classifier, fold, seed and protocol metrics |
| results/audited/cv_predictions2.csv | True labels, predictions, probabilities and row indices |
| results/audited/cv_mia2.csv | Membership-inference AUC per method/fold/seed |
| results/audited/heldout_fidelity.csv | Matched-size training and held-out KS summaries |
| results/example/ | Runnable classification and augmentation example outputs |
| results/sensitivity/ | Duplicate-CID sensitivity checks |
| results/original_study/ | Earlier saved evaluations, retained for provenance |
| figures/ | Eight figures in PDF, SVG and PNG, with a manuscript numbering index |
| code/ | Source retrieval, harmonisation, generation, training, evaluation and plotting |
| requirements.txt | Tested full-workflow dependency versions |
| requirements-core.txt | Lightweight example dependencies |
| CHECKSUMS.txt | SHA-256 manifest |

### Profile schema

The four synthetic CSVs and the real matrix have 80 columns:

- **Patient ID**: synthetic IDs use syn_&lt;class&gt;_&lt;n&gt;; real IDs use &lt;class&gt;_&lt;n&gt;.
  A synthetic ID identifies the same record in its class file and the combined file.
- **Disease**: asthma, bronchiectasis or copd.
- **78 numeric columns**: PubChem CIDs in ascending numerical order. Values are abundances
  on the source's arbitrary instrument scale, not molar concentrations.

The compound dictionary has its own two-column schema. Chemical assignments and names
are inherited from the source study. Harmonisation selects the first duplicate CID in
source-file order; two shared COPD CIDs are affected.

### Benchmark schema

Files named &lt;method&gt;_f&lt;fold&gt;_s&lt;seed&gt;.npz contain X (1,500 × 78 abundances) and y
(integer labels: 0=asthma, 1=bronchiectasis, 2=copd). Folds are 0–4 and seed indices 0–2;
numerical training seeds are 42 + fold + 1000 × seed_index.
JSON sidecars record partitions, feature order, parameters and environment.
Methods are wgangp, ctgan, tvae, copula and bootstrap; all have 500 profiles per class. The deterministic Gaussian copula uses one run
per fold (seed index 0); neural methods and bootstrap use all three seed indices.
SDV uses its fixed sampling stream (73251), so neural repetitions vary training initialisation.
Bootstrap samples contain resampled real training observations.

TSTR means train on synthetic, test on real; TRTR means train and test on separate real
partitions; TRTS means train on real, test on synthetic.
In prediction files, record_index refers to the real matrix for TSTR/TRTR and to
the generated sample for TRTS. p0, p1 and p2 are predicted class probabilities.
Standard deviations describe fold/seed variation, not independent-cohort confidence intervals.
Current manuscript figures use results/audited/, not results/original_study/.

## Regenerate the archived sample

~~~bash
python code/generate_release.py --verify
python code/generate_release.py --seed 123 --n-per-class 1000 --output-dir /tmp/breathnet-draw
~~~

The script loads the archived generator and **saved** preprocessing parameters.
Default generation writes to results/regenerated/, preserving the archived CSVs.
The release seed is 20260906. Verification compares scaled feature values at absolute
tolerance 1e-5 because floating-point kernels can vary across environments.
Checksums establish byte identity of archived files. Other seeds draw other samples
from the same fitted model.

## Rebuild the source matrix

~~~bash
python code/fetch_source.py --output-dir results/source
python code/build_real_matrix.py --source-dir results/source --verify
python code/duplicate_sensitivity.py --source-dir results/source
~~~

The downloader fixes figshare version 6 and checks source-file hashes. The rebuild
uses only the four documented source files and compares against the archived matrix
at relative and absolute tolerance 1e-12. Outputs go to results/rebuilt/.
Options --duplicate-policy last and mean support alternative source representations.
Sensitivity results describe source features and real-data logistic regression,
not retraining of the archived full-cohort generator.

Figure filenames retain their original numbering. See [figures/FIGURE_INDEX.md](figures/FIGURE_INDEX.md) for their locations in the revised manuscript and appendix.

## Reproduce validation and figures

The archived fold-specific samples support evaluation without generative retraining:

~~~bash
python code/fidelity.py
python code/noise_floor.py
python code/cv_eval2.py
python code/benchmark_example.py
python code/figures_schematic.py
python code/figures_fidelity.py
python code/figures_utility.py
~~~

Refit generators into a new directory to preserve archived benchmark samples:

~~~bash
python code/cv_splits.py
python code/cv_wgangp_seeds.py --output-dir results/rerun
python code/cv_baselines.py --output-dir results/rerun
python code/cv_eval2.py --input-dir results/rerun
python code/benchmark_example.py --input-dir results/rerun --output-dir results/rerun-example
~~~

Training is the expensive stage. TensorFlow and SDV/PyTorch execute in separate processes.
WGAN-GP uses 2,000 epochs and batch 32; CTGAN/TVAE use 1,000 epochs and batch 30.
Conditional sampling errors stop the run; exact class counts are asserted.
Metadata explicitly types VOCs as numerical and diagnosis as categorical.
The settings define a benchmark, not an exhaustive tuning search.

To train a new full-matrix model without replacing the archive:

~~~bash
python code/train_generator.py --output-dir /tmp/breathnet-new-model --seed 42
~~~

## Interpretation and reuse

- Additional draws increase resolution of the fitted model, not information about
  independent patients.
- Full-release fidelity measures agreement with the **training cohort**.
  Held-out comparisons evaluate independently refitted fold generators.
- Output is restricted to observed training ranges. The archived sample has compressed
  dispersion and imperfect compound dependence; account for these in downstream analyses.
- When testing on the 121 source patients, use fold-specific samples or refit after splitting.
  The full-cohort generator saw all 121 profiles.
- Labels may reflect source-study processing and population differences as well as biology.
  Random patient folds do not resolve these potential confounders.
- Membership inference is explicitly reported. Synthetic output and absence of exact copies
  do not establish privacy. The training procedure does not implement differential privacy.
- Generated records support methodological work, not diagnostic decisions, disease-mechanism
  claims or biomarker conclusions.

## Provenance, version and citation

Source: Kuo, P.-H. et al. *A Clinical Breathomics Dataset*. Scientific Data 11, 203 (2024).
[Article](https://doi.org/10.1038/s41597-024-03052-2).
[Peak tables, version 6](https://doi.org/10.6084/m9.figshare.23522490.v6), CC BY 4.0.

Release version: **2026-09-15**. Data and derived records:
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Code: MIT, as provided in LICENSE-CODE. Cite the source article and dataset alongside
this release. Reserved DOI: **10.5281/zenodo.21908977** (author-provided private Zenodo draft).
Public availability is pending publication of that record. See CITATION.cff for citation metadata.

Earlier public versions used “bronchitis” and called the model CTGAN.
The correct source condition is bronchiectasis, and the archived model is a
conditional WGAN-GP. CTGAN is a separate benchmark reference method.
