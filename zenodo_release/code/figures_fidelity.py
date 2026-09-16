"""Figures 2,3,4,5,8: training curves, marginals, correlation, PCA, privacy."""
import sys, json, numpy as np, pandas as pd, warnings

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from common import CLASSES, DATA, RESULTS, MODEL, FIGURES
from figure_style import apply_style, save_figure, DISEASE, DIVERGING
apply_style()

LBL = {'asthma':'Asthma','bronchiectasis':'Bronchiectasis','copd':'COPD'}
COL = DISEASE
FIG = FIGURES; FIG.mkdir(exist_ok=True); RES = RESULTS

real = pd.read_csv(DATA/'real_121x78.csv'); syn = pd.read_csv(DATA/'synthetic_release_6000.csv')
cols = list(real.columns[2:])
Xr, Xs = real[cols].values.astype(float), syn[cols].values.astype(float)
yr = pd.Categorical(real['Disease'],categories=CLASSES).codes
ys = pd.Categorical(syn['Disease'], categories=CLASSES).codes
ks  = pd.read_csv(RES/'ks_tests.csv', dtype={'pubchem_cid':str})
voc = pd.read_csv(DATA/'voc_dictionary.csv', dtype={'pubchem_cid':str}).set_index('pubchem_cid')['iupac_name'].to_dict()

def short(cid, n=26):
    s = str(voc.get(cid,'')).strip()
    return (s[:n-1]+'…') if len(s)>n else (s or f'CID {cid}')

# ---------------------------------------------------- Fig 2: marginals + KS
pooled = ks[ks.group=='pooled'].set_index('pubchem_cid')
order  = pooled['ks'].sort_values().index.tolist()
picks  = [order[0], order[len(order)//4], order[len(order)//2], order[3*len(order)//4], order[-1]]
fig = plt.figure(figsize=(7.1, 3.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1,0.95], hspace=0.95)
gs_top = gs[0].subgridspec(1, 5, wspace=0.42)
gs_bottom = gs[1].subgridspec(1, 2, width_ratios=[3,2], wspace=0.28)
for i, cid in enumerate(picks):
    ax = fig.add_subplot(gs_top[0,i])
    j  = cols.index(cid)
    lo = np.log10(np.maximum(np.r_[Xr[:,j],Xs[:,j]],1e-3))
    bins = np.linspace(lo.min(), lo.max(), 26)
    ax.hist(np.log10(np.maximum(Xs[:,j],1e-3)), bins=bins, density=True, alpha=.55,
            color='#7A5195', lw=0, label='Synthetic')
    ax.hist(np.log10(np.maximum(Xr[:,j],1e-3)), bins=bins, density=True, histtype='step',
            color='#263238', lw=.9, label='Real')
    ax.set_title(f"CID {cid}\n$D$ = {pooled.loc[cid,'ks']:.3f}", fontsize=7.2, pad=2.5)
    ax.set_xlabel(r'$\log_{10}$ abundance', fontsize=7.2)
    if i==0: ax.set_ylabel('Density', fontsize=7.5)
    ax.tick_params(labelsize=7.2)
    ax.set_yticks([])
    if i==0: ax.legend(fontsize=7.2, frameon=False, loc='upper left', handlelength=1.1)
axb = fig.add_subplot(gs_bottom[0,0])
for c, style in zip(CLASSES, ['-', '--', ':']):
    s = ks[ks.group==c]['ks'].values
    axb.hist(s, bins=np.linspace(0,.45,28), histtype='step', lw=1.2, ls=style, color=COL[c], label=LBL[c])
axb.axvline(ks[ks.group=='pooled']['ks'].median(), ls='--', lw=.8, c='#263238')
axb.set_xlabel(r'Kolmogorov--Smirnov statistic $D$', fontsize=8.0)
axb.set_ylabel('VOCs', fontsize=8.0); axb.legend(fontsize=7.2, frameon=False)
axb.tick_params(labelsize=7.2)
axc = fig.add_subplot(gs_bottom[0,1])
m = ks[ks.group=='pooled']
axc.scatter(np.log1p(Xr).mean(0), np.log1p(Xs).mean(0), s=4, c='#7A5195', lw=0, alpha=.8)
lims=[min(np.log1p(Xr).mean(0).min(),np.log1p(Xs).mean(0).min())-.4,
      max(np.log1p(Xr).mean(0).max(),np.log1p(Xs).mean(0).max())+.4]
axc.plot(lims,lims,'k--',lw=.7); axc.set_xlim(lims); axc.set_ylim(lims)
r = np.corrcoef(np.log1p(Xr).mean(0), np.log1p(Xs).mean(0))[0,1]
axc.set_xlabel(r'Real mean $\log(1+x)$', fontsize=8.0)
axc.set_ylabel(r'Synthetic mean $\log(1+x)$', fontsize=8.0)
axc.set_title(f'$r = {r:.4f}$', fontsize=8.0, pad=2)
axc.tick_params(labelsize=7.2)
fig.axes[0].text(-0.42,1.34,'a',transform=fig.axes[0].transAxes,fontsize=9.5,fontweight='bold',va='top')
for ax,l in zip([axb,axc],['b','c']):
    ax.text(-0.11 if l=='b' else 0,1.25,l,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='top')
save_figure(fig, FIG/'fig3_marginals.pdf'); plt.close(fig)
print('fig2 ok')

# ------------------------------------------------- Fig 3: correlation matrices
fig, axes = plt.subplots(1,3, figsize=(7.1,2.5), layout='constrained')
Cr = np.load(RES/'corr_real_pooled.npy'); Cs = np.load(RES/'corr_syn_pooled.npy')
for ax,(M,t,cm,vmin,vmax) in zip(axes, [(Cr,'Real ($n=121$)',DIVERGING,-1,1),
                                        (Cs,'Synthetic ($n=6000$)',DIVERGING,-1,1),
                                        (Cr-Cs,'Real − synthetic',DIVERGING,-.6,.6)]):
    im = ax.imshow(M, cmap=cm, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(t, fontsize=8.5, pad=3); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=7.2)
axes[0].set_ylabel('VOC', fontsize=8.0)
iu = np.triu_indices_from(Cr,k=1)
axes[2].set_xlabel(f"mean $|\\Delta\\rho|$ = {np.abs((Cr-Cs)[iu]).mean():.3f}", fontsize=7.5)
for ax,l in zip(axes,['a','b','c']):
    ax.text(-.02,1.13,l,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='top')
save_figure(fig, FIG/'fig4_correlation.pdf'); plt.close(fig)
print('fig3 ok')

# --------------------------------------------------------- Fig 4: PCA overlap
Pr, Ps = np.load(RES/'pca_real.npy'), np.load(RES/'pca_syn.npy')
evr = np.load(RES/'pca_evr.npy')
fig, axes = plt.subplots(1,3, figsize=(7.1,2.7), layout='constrained')
for ax,(a,b) in zip(axes[:2], [(0,1),(0,2)]):
    for ci,c in enumerate(CLASSES):
        ax.scatter(Ps[ys==ci,a],Ps[ys==ci,b],s=1.4,c=COL[c],alpha=.28,lw=0,rasterized=True)
    for ci,c in enumerate(CLASSES):
        ax.scatter(Pr[yr==ci,a],Pr[yr==ci,b],s=11,facecolor='none',edgecolor=COL[c],lw=.65)
    ax.set_xlabel(f'PC{a+1} ({evr[a]*100:.1f}%)', fontsize=8.0)
    ax.set_ylabel(f'PC{b+1} ({evr[b]*100:.1f}%)', fontsize=8.0)
    ax.tick_params(labelsize=7.2)
h = [Line2D([],[],marker='o',ls='',mfc=COL[c],mec=COL[c],ms=3,label=LBL[c]) for c in CLASSES]
h += [Line2D([],[],marker='o',ls='',mfc='none',mec='#263238',ms=3.4,label='Real'),
      Line2D([],[],marker='o',ls='',mfc='#263238',mec='#263238',ms=3,label='Synthetic')]
fig.legend(handles=h, fontsize=7.2, frameon=False, loc='outside upper center', ncol=5, handletextpad=.3)
ax = axes[2]
w=.36; x=np.arange(6)
ax.bar(x-w/2,[Pr[:,k].std() for k in range(6)],w,color='#263238',label='Real')
ax.bar(x+w/2,[Ps[:,k].std() for k in range(6)],w,color='#7A5195',label='Synthetic')
ax.set_xticks(x); ax.set_xticklabels([f'PC{k+1}' for k in range(6)],fontsize=7.2)
ax.set_ylabel('Standard deviation', fontsize=8.0); ax.legend(fontsize=7.2,frameon=False)
ax.tick_params(labelsize=7.2)
for ax,l in zip(axes,['a','b','c']):
    ax.text(-.06,1.1,l,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='top')
save_figure(fig, FIG/'fig5_pca.pdf'); plt.close(fig)
print('fig4 ok')

# ------------------------------------------------------ Fig 7: privacy / NN
d_syn, d_real = np.load(RES/'dcr_syn.npy'), np.load(RES/'dcr_real.npy')
priv = pd.read_csv(RES/'privacy.csv')
fig, axes = plt.subplots(1,3, figsize=(7.1,2.3), layout='constrained')
ax=axes[0]
hi=np.percentile(np.r_[d_syn,d_real],99.5); bins=np.linspace(0,hi,40)
ax.hist(d_syn,bins=bins,density=True,color='#7A5195',alpha=.6,lw=0,label='Synthetic $\\rightarrow$ real')
ax.hist(d_real,bins=bins,density=True,histtype='step',color='#263238',lw=1.0,label='Real $\\rightarrow$ real')
ax.axvline(np.median(d_syn),color='#7A5195',ls='--',lw=.8)
ax.axvline(np.median(d_real),color='#263238',ls='--',lw=.8)
ax.set_xlabel('Distance to closest record', fontsize=8.0); ax.set_ylabel('Density', fontsize=8.0)
ax.set_xlim(0,hi); ax.legend(fontsize=7.2, frameon=False); ax.tick_params(labelsize=7.2)
ax=axes[1]
xx=np.arange(3); p=priv[priv.group!='pooled']
ax.bar(xx-.19,p.dcr_median,.38,color='#7A5195',label='Synthetic $\\rightarrow$ real')
ax.bar(xx+.19,p.real_nn_median,.38,color='#263238',label='Real $\\rightarrow$ real')
ax.set_xticks(xx); ax.set_xticklabels([LBL[c] for c in CLASSES],fontsize=7.2)
ax.set_ylabel('Median NN distance', fontsize=8.0)
ax.tick_params(axis='x', rotation=25)
ax.set_ylim(0,8.4); ax.tick_params(labelsize=7.2)
ax=axes[2]
for ci,c in enumerate(CLASSES):
    sub = priv[priv.group==c]
    ax.bar(ci, sub.ratio_dcr_to_realnn.values[0], .55, color=COL[c])
ax.axhline(1.0, ls='--', c='#263238', lw=.8)
ax.set_xticks(range(3)); ax.set_xticklabels([LBL[c] for c in CLASSES],fontsize=7.2)
ax.set_ylabel('DCR / real-real NN', fontsize=8.0); ax.set_ylim(0,1.15)
ax.tick_params(axis='x', rotation=25)
ax.tick_params(labelsize=7.2)
for ax,l in zip(axes,['a','b','c']):
    ax.text(-.07,1.1,l,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='top')
save_figure(fig, FIG/'fig8_privacy.pdf'); plt.close(fig)
print('fig7 ok')

# --------------------------------------------------- Fig 8: training curves
h = json.loads((MODEL/'training_history.json').read_text())
fig, axes = plt.subplots(1,3, figsize=(7.1,2.0), layout='constrained')
for ax,(k,t,c) in zip(axes, [('d_loss','Critic loss','#0072B2'),
                             ('g_loss','Generator loss','#D55E00'),
                             ('gp','Gradient penalty','#009E73')]):
    v=np.asarray(h[k]); ax.plot(v,lw=.45,color=c,alpha=.75)
    w=25; ax.plot(np.arange(w-1,len(v)),np.convolve(v,np.ones(w)/w,'valid'),lw=.9,color='#263238')
    ax.set_xlabel('Epoch',fontsize=8.0); ax.set_ylabel(t,fontsize=8.0); ax.tick_params(labelsize=7.2)
    if k=='gp': ax.set_yscale('log')
for ax,l in zip(axes,['a','b','c']):
    ax.text(-.08,1.12,l,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='top')
save_figure(fig, FIG/'fig2_training.pdf'); plt.close(fig)
print('fig8 ok')
