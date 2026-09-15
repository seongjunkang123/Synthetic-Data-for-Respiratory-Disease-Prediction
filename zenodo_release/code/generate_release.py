"""Sample the archived generator using its saved transform, from any directory."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA, MODEL, RESULTS, CLASSES, Preprocessor, environment, validate_sample

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20260906)
    p.add_argument('--n-per-class', type=int, default=2000)
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'regenerated')
    p.add_argument('--verify', action='store_true', help='Compare with the archived 6000 rows')
    a = p.parse_args()
    if a.n_per_class < 1:
        p.error('--n-per-class must be positive')
    import wgangp
    import keras
    gen = keras.models.load_model(MODEL / 'generator.keras', compile=False)
    pre = Preprocessor.load(MODEL / 'preprocessing.json')
    cols = list(pd.read_csv(DATA / 'voc_dictionary.csv').pubchem_cid.astype(str))
    X, y = wgangp.sample(gen, pre, a.n_per_class, seed=a.seed)
    validate_sample(X, y, a.n_per_class)
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, 'Disease', [CLASSES[i] for i in y])
    df.insert(0, 'Patient ID', [f'syn_{c}_{j+1}' for c in CLASSES for j in range(a.n_per_class)])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    name = 'synthetic_release_6000.csv' if a.n_per_class == 2000 else 'synthetic_combined.csv'
    df.to_csv(a.output_dir / name, index=False)
    for c in CLASSES:
        df[df.Disease == c].to_csv(a.output_dir / f'synthetic_{c}.csv', index=False)
    info = dict(seed=a.seed, n_per_class=a.n_per_class, environment=environment())
    if a.verify:
        if a.seed != 20260906 or a.n_per_class != 2000:
            p.error('--verify requires the release seed and 2000 samples per class')
        ref = pd.read_csv(DATA / 'synthetic_release_6000.csv')
        assert list(ref.columns) == list(df.columns)
        assert ref.iloc[:, :2].equals(df.iloc[:, :2])
        delta = np.abs(pre.transform(ref[cols].to_numpy()) - pre.transform(X))
        info.update(max_absolute_scaled_error=float(delta.max()),
                    scaled_tolerance=1e-5, numeric_match=bool(delta.max() <= 1e-5))
    (a.output_dir / 'generation_report.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    if a.verify and not info['numeric_match']:
        raise AssertionError('Generated values exceed the stated cross-platform tolerance')

if __name__ == '__main__':
    main()
