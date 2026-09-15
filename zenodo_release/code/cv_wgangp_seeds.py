"""Refit and archive WGAN-GP within each patient fold and seed."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import RESULTS, validate_sample, environment
import cv_splits as S

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'audited')
    a = p.parse_args()
    import wgangp
    _, X, y, cols = S.load()
    out = a.output_dir / 'synth'
    out.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        for k, (tr, te) in enumerate(S.get_folds()):
            if k not in a.folds:
                continue
            dest = out / f'wgangp_f{k}_s{s}.npz'
            meta = dest.with_suffix('.json')
            if dest.exists() and meta.exists():
                saved = json.loads(meta.read_text())
                assert saved['epochs'] == a.epochs and saved['train_indices'] == tr
                sample = np.load(dest)
                validate_sample(sample['X'], sample['y'], 500)
                print(f'cached WGAN-GP fold {k} seed {s}', flush=True)
                continue
            t = time.monotonic()
            seed = 42 + k + 1000 * s
            gen, pre, hist = wgangp.train(X[tr], y[tr], epochs=a.epochs, seed=seed)
            Xs, ys = wgangp.sample(gen, pre, 500, seed=seed)
            validate_sample(Xs, ys, 500)
            np.savez_compressed(dest, X=Xs, y=ys)
            # Audit data bind each sample to its actual training partition and parameters.
            info = dict(method='wgangp', fold=k, seed_index=s, random_seed=seed,
                        epochs=a.epochs, batch_size=wgangp.BATCH_SIZE, latent_dimension=wgangp.LATENT_DIM,
                        critic_updates=wgangp.N_CRITIC, gradient_penalty_weight=wgangp.GP_WEIGHT,
                        learning_rate=wgangp.LR, adam_beta1=wgangp.BETA_1, adam_beta2=wgangp.BETA_2,
                        sampling_seed=seed, n_per_class=500, train_indices=tr,
                        test_indices=te, columns=cols, seconds=time.monotonic()-t,
                        environment=environment())
            meta.write_text(json.dumps(info, indent=2) + '\n')
            pre.save(out / f'wgangp_f{k}_s{s}_preprocessing.json')
            (out / f'wgangp_f{k}_s{s}_history.json').write_text(json.dumps(hist))
            print(f'WGAN-GP fold {k} seed {s}: {info["seconds"]:.1f}s', flush=True)
            import keras
            keras.backend.clear_session()

if __name__ == '__main__':
    main()
