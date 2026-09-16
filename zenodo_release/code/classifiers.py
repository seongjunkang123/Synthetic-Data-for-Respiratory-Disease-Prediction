"""Downstream classifiers used for the TSTR / TRTR / TRTS utility protocol."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def _log1p(X): return np.log1p(np.maximum(X, 0))

def make(name, seed=0):
    """All classifiers see log1p-transformed, standardised features.
    The scaler is fit ONLY on that classifier's training set."""
    pre = [('log', FunctionTransformer(_log1p)), ('sc', StandardScaler())]
    clf = {
        'logreg': LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
        'rf':     RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1),
        'xgb':    XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                random_state=seed, n_jobs=1, tree_method='hist',
                                objective='multi:softprob', num_class=3, verbosity=0),
        'mlp':    MLPClassifier(hidden_layer_sizes=(256,128), max_iter=1500,
                                early_stopping=True, n_iter_no_change=40,
                                random_state=seed),
    }[name]
    return Pipeline(pre + [('clf', clf)])

NAMES = ['logreg', 'rf', 'xgb', 'mlp']

def fit_predict(name, Xtr, ytr, Xte, seed=0):
    m = make(name, seed).fit(Xtr, ytr)
    return m.predict(Xte), m.predict_proba(Xte)

def score(y_true, y_pred, y_prob):
    out = {'accuracy': accuracy_score(y_true, y_pred),
           'macro_f1': f1_score(y_true, y_pred, average='macro')}
    out['macro_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                     average='macro', labels=[0,1,2])
    return out
