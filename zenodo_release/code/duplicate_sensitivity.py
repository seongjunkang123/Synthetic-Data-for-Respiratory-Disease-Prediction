"""Assess the effect of first/last/mean duplicate CID resolution on source profiles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits
from common import RESULTS
from build_real_matrix import build
import cv_splits as S

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    a = p.parse_args()
    out = RESULTS / 'sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    df, ref, y, cols = S.load()
    summaries, runs = [], []
    with threadpool_limits(limits=2):
        for policy in ['first', 'last', 'mean']:
            matrix, _, _ = build(a.source_dir, policy)
            X = matrix.iloc[:, 2:].to_numpy(dtype=float)
            for j, cid in enumerate(cols):
                if not np.allclose(X[:, j], ref[:, j], rtol=1e-12, atol=1e-12):
                    delta = np.abs(np.log1p(X[:, j]) - np.log1p(ref[:, j]))
                    summaries.append(dict(policy=policy, cid=cid, n_changed=int((delta > 1e-10).sum()),
                                          median_abs_log_change=float(np.median(delta[y == 2])),
                                          max_abs_log_change=float(delta.max())))
            for k, (tr, te) in enumerate(S.get_folds()):
                m = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                  LogisticRegression(C=1, max_iter=5000, random_state=42)).fit(X[tr], y[tr])
                runs.append(dict(policy=policy, fold=k, accuracy=accuracy_score(y[te], m.predict(X[te]))))
    pd.DataFrame(summaries).to_csv(out / 'duplicate_changes.csv', index=False)
    pd.DataFrame(runs).to_csv(out / 'duplicate_cv.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(runs).groupby('policy').accuracy.agg(['mean', 'std']).to_string())
