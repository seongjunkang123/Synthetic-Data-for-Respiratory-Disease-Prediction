"""Evaluate audited synthetic sets on untouched real folds; archive predictions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from common import RESULTS, CLASSES, Preprocessor, validate_sample, environment
import cv_splits as S
import classifiers as C

def bootstrap(X, y, seed, n=500):
    rng = np.random.default_rng(seed)
    return (np.vstack([X[rng.choice(np.flatnonzero(y == c), n, replace=True)] for c in range(3)]),
            np.repeat(np.arange(3), n))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=RESULTS / 'audited')
    p.add_argument('--methods', nargs='+', default=['wgangp', 'ctgan', 'tvae', 'copula', 'bootstrap'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--wait', action='store_true', help='Wait for concurrently generated audited samples')
    a = p.parse_args()
    _, X, y, cols = S.load()
    F = S.get_folds()
    out = a.input_dir
    out.mkdir(exist_ok=True, parents=True)
    rows, preds, mia, fidelity = [], [], [], []
    def evaluate(method, seed, fold, proto, Xtr, ytr, Xte, yte, ids):
        for clf in C.NAMES:
            yp, prob = C.fit_predict(clf, Xtr, ytr, Xte, seed=42)
            base = dict(fold=fold, synth=method, seed=seed, protocol=proto, clf=clf)
            rows.append(dict(**base, **C.score(yte, yp, prob)))
            preds.extend(dict(**base, record_index=int(idx), true=int(t), pred=int(pr),
                              p0=float(pb[0]), p1=float(pb[1]), p2=float(pb[2]))
                         for idx, t, pr, pb in zip(ids, yte, yp, prob))
    with threadpool_limits(limits=2):
        for k, (tr, te) in enumerate(F):
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            evaluate('real', 0, k, 'TRTR', Xtr, ytr, Xte, yte, te)
            pre = Preprocessor().fit(Xtr)
            sd = pre.transform(Xtr).std(0) + 1e-12
            for method in a.methods:
                for seed in a.seeds:
                    if method == 'copula' and seed != 0:
                        continue
                    t = time.monotonic()
                    if method == 'bootstrap':
                        Xs, ys = bootstrap(Xtr, ytr, 42 + k + 1000 * seed)
                        (out / 'synth').mkdir(exist_ok=True)
                        np.savez_compressed(out / 'synth' / f'bootstrap_f{k}_s{seed}.npz', X=Xs, y=ys)
                        (out / 'synth' / f'bootstrap_f{k}_s{seed}.json').write_text(json.dumps(dict(
                            method=method, fold=k, seed_index=seed, random_seed=42+k+1000*seed,
                            train_indices=tr, test_indices=te, columns=cols, n_per_class=500,
                            sampling='class-wise resampling with replacement', environment=environment()), indent=2))
                    else:
                        path = out / 'synth' / f'{method}_f{k}_s{seed}.npz'
                        if a.wait:
                            started = time.monotonic()
                            while not path.with_suffix('.json').exists():
                                if time.monotonic() - started > 10800:
                                    raise TimeoutError(f'Missing audited sample: {path}')
                                time.sleep(5)
                        meta = json.loads(path.with_suffix('.json').read_text())
                        assert meta['train_indices'] == tr and meta['test_indices'] == te
                        d = np.load(path)
                        Xs, ys = d['X'], d['y']
                    validate_sample(Xs, ys, 500)
                    evaluate(method, seed, k, 'TSTR', Xs, ys, Xte, yte, te)
                    evaluate(method, seed, k, 'TRTS', Xtr, ytr, Xs, ys, np.arange(len(Xs)))
                    # Distance scoring uses preprocessing learned on members only.
                    nn = NearestNeighbors(n_neighbors=1).fit(pre.transform(Xs) / sd)
                    dm = nn.kneighbors(pre.transform(Xtr) / sd)[0][:, 0]
                    dn = nn.kneighbors(pre.transform(Xte) / sd)[0][:, 0]
                    mia.append(dict(fold=k, synth=method, seed=seed, mia_auc=roc_auc_score(
                        np.r_[np.ones(len(dm)), np.zeros(len(dn))], -np.r_[dm, dn])))
                    # Test and training comparisons use equally sized synthetic and real draws.
                    rng = np.random.default_rng(9000 + 1000 * seed + k)
                    for c in range(3):
                        T, R, G = Xte[yte == c], Xtr[ytr == c], Xs[ys == c]
                        n = len(T)
                        vals = []
                        for rep in range(30):
                            sample = G[rng.choice(len(G), n, replace=False)]
                            train = R[rng.choice(len(R), n, replace=False)]
                            for j, cid in enumerate(cols):
                                vals.append((ks_2samp(T[:, j], sample[:, j], method='asymp').statistic,
                                             ks_2samp(train[:, j], sample[:, j], method='asymp').statistic))
                        vals = np.asarray(vals).reshape(30, len(cols), 2)
                        fidelity.append(dict(fold=k, synth=method, seed=seed, disease=CLASSES[c],
                            n_real_test=n, resamples=30,
                            heldout_ks=float(np.median(vals[:, :, 0], axis=1).mean()),
                            training_ks=float(np.median(vals[:, :, 1], axis=1).mean())))
                    print(f'evaluated {method} fold {k} seed {seed}: {time.monotonic()-t:.1f}s', flush=True)
            pd.DataFrame(rows).to_csv(out / 'cv_results2.csv', index=False)
            pd.DataFrame(preds).to_csv(out / 'cv_predictions2.csv', index=False)
            pd.DataFrame(mia).to_csv(out / 'cv_mia2.csv', index=False)
            pd.DataFrame(fidelity).to_csv(out / 'heldout_fidelity.csv', index=False)
    (out / 'evaluation_environment.json').write_text(json.dumps(environment(), indent=2) + '\n')
    (out / 'COMPLETE.json').write_text(json.dumps(dict(methods=a.methods, seeds=a.seeds, folds=5)) + '\n')

if __name__ == '__main__':
    main()
