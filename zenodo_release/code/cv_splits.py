"""Fixed patient-level stratified five-fold splits, shared by every method."""
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import CLASSES, RESULTS, load_real
N_SPLITS, SEED = 5, 42
load = load_real

def folds(X, y):
    return [(tr.tolist(), te.tolist()) for tr, te in
            StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y)]

def get_folds():
    _, X, y, _ = load()
    expected = folds(X, y)
    path = RESULTS / 'folds.json'
    if path.exists():
        assert json.loads(path.read_text()) == [[tr, te] for tr, te in expected]
    return expected

if __name__ == '__main__':
    RESULTS.mkdir(exist_ok=True)
    _, X, y, _ = load()
    f = folds(X, y)
    (RESULTS / 'folds.json').write_text(json.dumps(f, indent=2) + '\n')
    for k, (tr, te) in enumerate(f):
        print(f'fold {k}: train={len(tr)} test={len(te)} classes={np.bincount(y[te]).tolist()}')
