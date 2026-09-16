"""Train a new full-matrix generator without overwriting the archived model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA, CLASSES, environment

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=DATA/'real_121x78.csv')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=2000)
    a = p.parse_args()
    import wgangp
    df = pd.read_csv(a.data)
    X = df.iloc[:,2:].to_numpy(float)
    y = pd.Categorical(df.Disease,categories=CLASSES).codes
    if (y<0).any() or not np.isfinite(X).all() or (X<0).any():
        raise ValueError('Expected non-negative finite abundances and the three supported labels')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError('Choose an empty output directory for a newly trained model')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    gen, pre, history = wgangp.train(X,y,epochs=a.epochs,seed=a.seed)
    gen.save(a.output_dir/'generator.keras')
    pre.save(a.output_dir/'preprocessing.json')
    (a.output_dir/'training_history.json').write_text(json.dumps(history))
    (a.output_dir/'training_run.json').write_text(json.dumps(dict(
        seed=a.seed,epochs=a.epochs,n_profiles=len(df),columns=list(df.columns[2:]),
        environment=environment()),indent=2)+'\n')
