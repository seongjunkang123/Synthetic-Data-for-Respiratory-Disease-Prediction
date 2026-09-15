"""A small reproducible reuse example: real-only, bootstrap and WGAN augmentation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits
from common import RESULTS, environment
import cv_splits as S

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=RESULTS / 'audited')
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'example')
    p.add_argument('--bootstrap-only', action='store_true', help='Run without fold-specific WGAN samples')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    _, X, y, _ = S.load()
    rows = []
    with threadpool_limits(limits=2):
        for k, (tr, te) in enumerate(S.get_folds()):
            Xtr, ytr = X[tr], y[tr]
            for s in [0, 1, 2]:
                rng = np.random.default_rng(42 + k + 1000*s)
                ix = np.concatenate([rng.choice(np.flatnonzero(ytr == c), 500, replace=True)
                                     for c in range(3)])
                sets = [('real_only', Xtr, ytr), ('bootstrap_only', Xtr[ix], ytr[ix])]
                if not a.bootstrap_only:
                    path = a.input_dir / 'synth' / f'wgangp_f{k}_s{s}.npz'
                    meta = json.loads(path.with_suffix('.json').read_text())
                    assert meta['train_indices'] == tr and meta['test_indices'] == te
                    d = np.load(path)
                    sets += [('synthetic_only', d['X'], d['y']),
                             ('real_plus_synthetic', np.vstack([Xtr, d['X']]), np.r_[ytr, d['y']])]
                for name, A, b in sets:
                    if name == 'real_only' and s:
                        continue
                    model = make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                          LogisticRegression(C=1, max_iter=5000, random_state=42))
                    model.fit(A, b)
                    pred, prob = model.predict(X[te]), model.predict_proba(X[te])
                    rows.append(dict(fold=k, seed=s, training=name, n_train=len(A),
                        accuracy=accuracy_score(y[te], pred),
                        macro_f1=f1_score(y[te], pred, average='macro'),
                        macro_auc=roc_auc_score(y[te], prob, multi_class='ovr', average='macro')))
    result = pd.DataFrame(rows)
    result.to_csv(a.output_dir / 'benchmark_runs.csv', index=False)
    summary = result.groupby('training').accuracy.agg(['mean', 'std', 'count'])
    summary.to_csv(a.output_dir / 'benchmark_summary.csv')
    (a.output_dir / 'environment.json').write_text(json.dumps(environment(), indent=2) + '\n')
    print(summary.to_string())

if __name__ == '__main__':
    main()
