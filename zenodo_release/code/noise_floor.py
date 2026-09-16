"""
Real-vs-real reference distributions for the fidelity metrics.

A raw synthetic-vs-real distance is hard to read on its own: with 33 to 53 real
profiles per class, two disjoint halves of the REAL cohort already disagree
substantially. This script measures that disagreement and uses it as the
reference against which the synthetic-vs-real distance is judged.

Design: every comparison uses the same two sample sizes, so the statistics are
directly comparable.
  reference   half A of real   vs   half B of real   (disjoint)
  synthetic   half A of real   vs   a synthetic draw of the same size as half B
Correlation uses full-class bootstrap resamples. These summaries describe
variation within the training cohort and are not independent-sample equivalence tests.
"""
import sys, numpy as np, pandas as pd, warnings

from scipy import stats
from pathlib import Path
from common import CLASSES, DATA, RESULTS
RES = RESULTS; RNG = np.random.default_rng(7)
NREP = 300

real = pd.read_csv(DATA/'real_121x78.csv'); syn = pd.read_csv(DATA/'synthetic_release_6000.csv')
cols = list(real.columns[2:])
Xr, Xs = real[cols].values.astype(float), syn[cols].values.astype(float)
yr = pd.Categorical(real['Disease'], categories=CLASSES).codes
ys = pd.Categorical(syn['Disease'],  categories=CLASSES).codes

def ks_med(A, B):
    return float(np.median(stats.ks_2samp(A, B, axis=0, method='asymp').statistic))

def w1_med(A, B):
    """Median per-feature 1-Wasserstein distance on the log scale, in units of
    the real per-feature SD, so it is comparable across compounds."""
    la, lb = np.log1p(A), np.log1p(B)
    sd = np.log1p(Xr).std(0) + 1e-12
    return float(np.median([stats.wasserstein_distance(la[:,j], lb[:,j])/sd[j]
                            for j in range(A.shape[1])]))

def corr_gap(A, B):
    Ca, Cb = stats.spearmanr(A).statistic, stats.spearmanr(B).statistic
    iu = np.triu_indices_from(Ca, k=1)
    return float(np.abs(Ca[iu]-Cb[iu]).mean())

rows = []
for ci, cname in enumerate(CLASSES):
    mr, ms = yr==ci, ys==ci
    R, S = Xr[mr], Xs[ms]
    n = len(R); h = n//2
    ref_ks, ref_w1, syn_ks, syn_w1 = [], [], [], []
    for _ in range(NREP):
        perm = RNG.permutation(n)
        A, B = R[perm[:h]], R[perm[h:2*h]]              # disjoint real halves
        Sd = S[RNG.choice(len(S), size=h, replace=False)]  # synthetic, same size
        ref_ks.append(ks_med(A,B));  ref_w1.append(w1_med(A,B))
        syn_ks.append(ks_med(A,Sd)); syn_w1.append(w1_med(A,Sd))
    ref_c, syn_c = [], []
    for _ in range(NREP):
        A = R[RNG.choice(n, size=n, replace=True)]
        B = R[RNG.choice(n, size=n, replace=True)]
        ref_c.append(corr_gap(A,B))
        syn_c.append(corr_gap(A, S[RNG.choice(len(S), size=n, replace=False)]))
    def q(v): v=np.asarray(v); return v.mean(), np.percentile(v,2.5), np.percentile(v,97.5)
    for metric, ref, syn_ in [('ks_median',ref_ks,syn_ks), ('w1_median_sd',ref_w1,syn_w1),
                              ('corr_mean_abs_diff',ref_c,syn_c)]:
        rm,rl,rh = q(ref); sm,sl,sh = q(syn_)
        rows.append(dict(group=cname, metric=metric, n_real=n,
                         real_vs_real=rm, rvr_lo=rl, rvr_hi=rh,
                         syn_vs_real=sm, svr_lo=sl, svr_hi=sh,
                         ratio=sm/rm if rm else np.nan))
df = pd.DataFrame(rows); df.to_csv(RES/'noise_floor.csv', index=False)
pd.set_option('display.width',200)
print("Real-vs-real reference vs synthetic-vs-real, matched sample sizes, "
      f"{NREP} resamples\n")
for m in ['ks_median','w1_median_sd','corr_mean_abs_diff']:
    print(f"--- {m} ---")
    for _,r in df[df.metric==m].iterrows():
        print(f"  {r.group:16s} n={int(r.n_real):3d}  real-real {r.real_vs_real:.3f} "
              f"[{r.rvr_lo:.3f},{r.rvr_hi:.3f}]   syn-real {r.syn_vs_real:.3f} "
              f"[{r.svr_lo:.3f},{r.svr_hi:.3f}]   ratio {r.ratio:.2f}")
    print()
