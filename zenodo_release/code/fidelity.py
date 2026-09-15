"""Descriptive agreement of the archived synthetic release with its training cohort."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from common import DATA, RESULTS, CLASSES, Preprocessor

def main():
    RESULTS.mkdir(exist_ok=True)
    real = pd.read_csv(DATA / 'real_121x78.csv')
    syn = pd.read_csv(DATA / 'synthetic_release_6000.csv')
    cols = list(real.columns[2:])
    Xr, Xs = real[cols].to_numpy(float), syn[cols].to_numpy(float)
    yr = pd.Categorical(real.Disease, categories=CLASSES).codes
    ys = pd.Categorical(syn.Disease, categories=CLASSES).codes
    rows, cor, privacy = [], [], []
    pre = Preprocessor().fit(Xr)
    Zr, Zs = pre.transform(Xr), pre.transform(Xs)
    sd = Zr.std(0) + 1e-12
    Ar, As = (Zr-Zr.mean(0))/sd, (Zs-Zr.mean(0))/sd
    for ci, c in enumerate([*CLASSES, 'pooled']):
        mr = np.ones(len(Xr), bool) if c == 'pooled' else yr == ci
        ms = np.ones(len(Xs), bool) if c == 'pooled' else ys == ci
        A, B = Xr[mr], Xs[ms]
        # Only effect sizes are used: synthetic and real samples are dependent.
        D = stats.ks_2samp(A, B, axis=0).statistic
        rows.extend(dict(group=c, pubchem_cid=cid, n_real=len(A), n_syn=len(B), ks=float(v))
                    for cid, v in zip(cols, D))
        Cr, Cs = stats.spearmanr(A).statistic, stats.spearmanr(B).statistic
        iu = np.triu_indices(78, 1)
        delta = Cr-Cs
        cor.append(dict(group=c, mean_abs_diff=float(np.abs(delta[iu]).mean()),
                        max_abs_diff=float(np.abs(delta[iu]).max()),
                        corr_of_corrs=float(np.corrcoef(Cr[iu], Cs[iu])[0, 1])))
        np.save(RESULTS / f'corr_real_{c}.npy', Cr)
        np.save(RESULTS / f'corr_syn_{c}.npy', Cs)
        R, S = Ar[mr], As[ms]
        nearest = NearestNeighbors(n_neighbors=2).fit(R)
        dd = nearest.kneighbors(S)[0]
        dr = nearest.kneighbors(R)[0][:, 1]
        ds = dd[:, 0]
        privacy.append(dict(group=c, n_real=len(R), n_syn=len(S),
            dcr_median=float(np.median(ds)), dcr_p5=float(np.percentile(ds, 5)),
            dcr_min=float(ds.min()), real_nn_median=float(np.median(dr)),
            real_nn_p5=float(np.percentile(dr, 5)),
            ratio_dcr_to_realnn=float(np.median(ds)/np.median(dr)),
            nndr_median=float(np.median(dd[:, 0]/np.maximum(dd[:, 1], 1e-12))),
            frac_closer_than_real_p5=float((ds < np.percentile(dr, 5)).mean()),
            exact_copies=int((ds < 1e-8).sum())))
        if c == 'pooled':
            np.save(RESULTS/'dcr_syn.npy', ds)
            np.save(RESULTS/'dcr_real.npy', dr)
    pd.DataFrame(rows).to_csv(RESULTS/'ks_tests.csv', index=False)
    pd.DataFrame(cor).to_csv(RESULTS/'correlation_structure.csv', index=False)
    pd.DataFrame(privacy).to_csv(RESULTS/'privacy.csv', index=False)
    pca = PCA(n_components=10).fit(Zr)
    for name, values in [('pca_real', pca.transform(Zr)), ('pca_syn', pca.transform(Zs)),
                         ('pca_evr', pca.explained_variance_ratio_)]:
        np.save(RESULTS/f'{name}.npy', values)
    m = pd.DataFrame(dict(pubchem_cid=cols, real_logmean=np.log1p(Xr).mean(0),
        syn_logmean=np.log1p(Xs).mean(0), real_logsd=np.log1p(Xr).std(0),
        syn_logsd=np.log1p(Xs).std(0)))
    m['sd_ratio'] = m.syn_logsd/m.real_logsd
    m.to_csv(RESULTS/'moments.csv', index=False)
    print('Archived-release fidelity, PCA and proximity statistics regenerated.')

if __name__ == '__main__':
    main()
