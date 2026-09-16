"""Paths, data schema and preprocessing shared by the release commands."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == 'analysis' and (ROOT.parent / 'zenodo_release').is_dir():
    ROOT = ROOT.parent / 'zenodo_release'
DATA, MODEL, RESULTS, FIGURES = [ROOT / p for p in ('data', 'model', 'results', 'figures')]
CLASSES = ['asthma', 'bronchiectasis', 'copd']

def load_real():
    df = pd.read_csv(DATA / 'real_121x78.csv')
    cols = list(df.columns[2:])
    X = df[cols].to_numpy(dtype=float)
    y = pd.Categorical(df.Disease, categories=CLASSES).codes.astype(int)
    assert X.shape == (121, 78) and np.isfinite(X).all() and (X >= 0).all()
    assert np.array_equal(np.bincount(y), [53, 35, 33])
    return df, X, y, cols

def validate_sample(X, y, n_per_class):
    assert X.shape == (3 * n_per_class, 78), X.shape
    assert y.shape == (len(X),) and set(y) == {0, 1, 2}
    assert np.isfinite(X).all() and (X >= 0).all()
    assert np.array_equal(np.bincount(y, minlength=3), [n_per_class] * 3)

class Preprocessor:
    """log1p and feature-wise min-max scaling, with saved-parameter support."""
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X):
        self.scaler.fit(np.log1p(X))
        return self

    def transform(self, X):
        return self.scaler.transform(np.log1p(X))

    def inverse(self, Z):
        return np.maximum(np.expm1(self.scaler.inverse_transform(Z)), 0.0)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.get('log_transform') is not True:
            raise ValueError('Expected log1p preprocessing')
        obj = cls()
        s = obj.scaler
        s.data_min_ = np.asarray(d['scaler_min'], dtype=float)
        s.data_max_ = np.asarray(d['scaler_max'], dtype=float)
        s.data_range_ = s.data_max_ - s.data_min_
        s.scale_ = np.asarray(d['scaler_scale'], dtype=float)
        s.min_ = -s.data_min_ * s.scale_
        s.n_features_in_ = len(s.scale_)
        return obj

    def save(self, path):
        s = self.scaler
        Path(path).write_text(json.dumps(dict(log_transform=True,
            scaler_min=s.data_min_.tolist(), scaler_max=s.data_max_.tolist(),
            scaler_scale=s.scale_.tolist()), indent=2) + '\n')

def environment():
    import platform
    from importlib.metadata import version, PackageNotFoundError
    d = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow',
                 'keras', 'torch', 'sdv', 'ctgan', 'xgboost', 'matplotlib']:
        try:
            d[name] = version(name)
        except PackageNotFoundError:
            pass
    return d


def release_files():
    """Yield archived files, excluding local environments and temporary rebuilds."""
    import os
    excluded = {'__pycache__', '.git', '.venv', 'regenerated', 'source', 'rebuilt'}
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(names):
            if name not in {'CHECKSUMS.txt', '.DS_Store'}:
                yield Path(base) / name
