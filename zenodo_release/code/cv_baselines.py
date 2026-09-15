"""Audited conditional SDV baselines, run separately from TensorFlow."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import CLASSES, RESULTS, validate_sample, environment
import cv_splits as S

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=['ctgan', 'tvae', 'copula'],
                   default=['ctgan', 'tvae', 'copula'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=30)
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'audited')
    a = p.parse_args()
    if a.batch_size % 10:
        p.error('CTGAN batch size must be divisible by packing factor 10')
    import torch
    torch.set_num_threads(2)
    from sdv.metadata import Metadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
    from sdv.sampling import Condition
    _, X, y, cols = S.load()
    out = a.output_dir / 'synth'
    out.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        for k, (tr, te) in enumerate(S.get_folds()):
            if k not in a.folds:
                continue
            d = pd.DataFrame(X[tr], columns=cols)
            d['Disease'] = [CLASSES[i] for i in y[tr]]
            md = Metadata.detect_from_dataframe(d, table_name='t', infer_sdtypes=False, infer_keys=None)
            for col in cols:
                md.update_column(table_name='t', column_name=col, sdtype='numerical')
            md.update_column(table_name='t', column_name='Disease', sdtype='categorical')
            for kind in a.methods:
                if kind == 'copula' and s != 0:
                    continue  # Deterministic fit and default sampling: one run per fold.
                dest = out / f'{kind}_f{k}_s{s}.npz'
                meta = dest.with_suffix('.json')
                if dest.exists() and meta.exists():
                    saved = json.loads(meta.read_text())
                    assert saved['epochs'] == a.epochs and saved['batch_size'] == a.batch_size
                    sample = np.load(dest)
                    validate_sample(sample['X'], sample['y'], 500)
                    print(f'cached {kind} fold {k} seed {s}', flush=True)
                    continue
                t = time.monotonic()
                seed = 42 + k + 1000*s
                import random
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                kw = dict(enforce_min_max_values=True, enforce_rounding=False)
                if kind == 'copula':
                    model = GaussianCopulaSynthesizer(md, **kw)
                else:
                    cls = CTGANSynthesizer if kind == 'ctgan' else TVAESynthesizer
                    model = cls(md, epochs=a.epochs, batch_size=a.batch_size,
                                verbose=False, cuda=False, **kw)
                model.fit(d)
                model.reset_sampling()
                conds = [Condition(num_rows=500, column_values={'Disease': c}) for c in CLASSES]
                # Failure is fatal: never substitute unconditional or unbalanced data.
                sampled = model.sample_from_conditions(conds, batch_size=1000)
                ys = pd.Categorical(sampled.Disease, categories=CLASSES).codes.astype(int)
                Xs = sampled[cols].to_numpy(dtype=float)
                validate_sample(Xs, ys, 500)
                np.savez_compressed(dest, X=Xs, y=ys)
                info = dict(method=kind, fold=k, seed_index=s, random_seed=seed,
                            epochs=a.epochs, batch_size=a.batch_size, n_per_class=500,
                            train_indices=tr, test_indices=te, columns=cols,
                            sampling='conditional; exact class counts asserted', sampling_seed=73251,
                            seconds=time.monotonic()-t, environment=environment())
                meta.write_text(json.dumps(info, indent=2) + '\n')
                md.save_to_json(str(out / f'{kind}_f{k}_s{s}_metadata.json'), mode='overwrite')
                print(f'{kind} fold {k} seed {s}: {info["seconds"]:.1f}s', flush=True)

if __name__ == '__main__':
    main()
