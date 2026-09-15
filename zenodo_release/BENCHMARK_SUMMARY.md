# Audited benchmark summary

Metrics are averages across four classifiers within each fold/seed run. Values are mean ± standard deviation over 15 runs per stochastic method and five copula folds.

| Method | TSTR accuracy | TSTR macro-AUC | Membership AUC |
|---|---:|---:|---:|
| wgangp | 0.844 ± 0.078 | 0.966 ± 0.015 | 0.671 ± 0.084 |
| ctgan | 0.341 ± 0.087 | 0.484 ± 0.112 | 0.521 ± 0.061 |
| tvae | 0.837 ± 0.050 | 0.942 ± 0.029 | 0.783 ± 0.042 |
| copula | 0.492 ± 0.064 | 0.627 ± 0.033 | 0.610 ± 0.058 |
| bootstrap | 0.914 ± 0.024 | 0.976 ± 0.010 | 1.000 ± 0.000 |

Real-only reference accuracy: 0.915 ± 0.021 over five folds.

## Runnable logistic-regression example

| Training condition | Accuracy | Runs |
|---|---:|---:|
| bootstrap_only | 0.959 ± 0.047 | 15 |
| real_only | 0.959 ± 0.051 | 5 |
| real_plus_synthetic | 0.942 ± 0.054 | 15 |
| synthetic_only | 0.878 ± 0.080 | 15 |

The bootstrap resamples real training records and does not generate new continuous profiles. All results describe this cohort and these configurations; variation across repeated folds is not an independent-cohort confidence interval.
