# Synthetic Data for Respiratory Disease Prediction

This directory contains synthetically generated Volatile Organic Compound (VOC) profiles for three respiratory diseases. The data was produced using the trained CTGAN (Conditional Tabular Generative Adversarial Network) from the [`sip_model_gen`](https://github.com/seongjunkang123/sip_model_gen) repository.
Refer to [sip_model_gen](https://github.com/seongjunkang123/sip_model_gen) [sip_data](https://github.com/seongjunkang123/sip_data) for more information. 
## Files

| File | Disease | Samples |
|------|---------|---------|
| `synthetic_data_asthma.csv` | Asthma | 500     |
| `synthetic_data_bronchi.csv` | Bronchitis | 500     |
| `synthetic_data_copd.csv` | COPD | 500     |

## Data Format

Each CSV file shares the same schema with 80 columns:

- **Patient ID** — Synthetic patient identifier (e.g., `asthma_1`, `bronchi_23`, `copd_50`).
- **Disease** — Disease label (`asthma`, `bronchi`, or `copd`).
- **78 VOC feature columns** — Each remaining column is named by its compound identifier and contains a floating-point concentration value representing a VOC measurement.

## How the Data Was Generated

The synthetic data was generated using `generate_synthetic_data.py` in the `sip_model_gen` repository. The generation process works as follows:

1. A MinMaxScaler is fit on the original training data (with optional log transformation) to replicate the same preprocessing used during model training.
2. The trained generator model is loaded from saved weights.
3. For each disease condition, 50 samples are produced by feeding random latent vectors (dimension 128) along with a one-hot encoded disease label into the generator.
4. The generator output is inverse-transformed back to the original data scale, and any negative values are clipped to zero since VOC concentrations cannot be negative.

## Related Repositories

- [`sip_model_gen`](https://github.com/seongjunkang123/sip_model_gen) — CTGAN model training and synthetic data generation.
- [`sip_model_class`](https://github.com/seongjunkang123/sip_model_class) — Classification model for respiratory disease prediction.
- [`sip_data`](https://github.com/seongjunkang123/sip_data) — Original VOC datasets.
