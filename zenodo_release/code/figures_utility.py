"""Figures 6 and 7 from the audited evaluation tables."""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RESULTS, FIGURES

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=str, default=str(RESULTS/'audited'))
    a = p.parse_args()
    from pathlib import Path
    root = Path(a.input_dir)
    R = pd.read_csv(root/'cv_results2.csv')
    M = pd.read_csv(root/'cv_mia2.csv')
    P = pd.read_csv(root/'cv_predictions2.csv')
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8,
                         'xtick.labelsize':7, 'ytick.labelsize':7, 'pdf.fonttype':42})
    names = ['logreg','rf','xgb','mlp']
    labels = ['Logistic','Random forest','XGBoost','MLP']
    purple, ink, blue = '#79568c','#30343b','#357da8'
    fig, axes = plt.subplots(2,2,figsize=(7.1,5.2),layout='constrained')
    x=np.arange(4)
    for offset, proto, method, label, color in [(-.19,'TRTR','real','Real training',ink),
                                               (.19,'TSTR','wgangp','Synthetic training',purple)]:
        q=R[(R.protocol==proto)&(R.synth==method)]
        means=[q[q.clf==n].accuracy.mean() for n in names]
        sds=[q[q.clf==n].accuracy.std() for n in names]
        axes[0,0].bar(x+offset,means,.38,yerr=sds,capsize=2,color=color,label=label)
    ax=axes[0,0];ax.set(xticks=x,xticklabels=labels,ylabel='Held-out real accuracy',ylim=(0,1.06))
    ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.2),ncol=2)
    ax.axhline(1/3,ls=':',color='gray',lw=.8)
    q=P[(P.protocol=='TSTR')&(P.synth=='wgangp')]
    cm=pd.crosstab(q.true,q.pred).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).to_numpy()
    cm=cm/cm.sum(axis=1,keepdims=True);ax=axes[0,1];ax.imshow(cm,cmap='Purples',vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',color='white' if cm[i,j]>.6 else ink,fontsize=9)
    diseases=['Asthma','Bronchiectasis','COPD']
    ax.set(xticks=range(3),yticks=range(3),xticklabels=diseases,yticklabels=diseases,xlabel='Predicted',ylabel='True')
    ax.tick_params(axis='x',rotation=25)
    q=R[(R.protocol=='TRTS')&(R.synth=='wgangp')]
    ax=axes[1,0];ax.bar(x,[q[q.clf==n].accuracy.mean() for n in names],.6,
        yerr=[q[q.clf==n].accuracy.std() for n in names],capsize=2,color=blue)
    ax.set(xticks=x,xticklabels=labels,ylabel='Synthetic test accuracy',ylim=(0,1.06),title='Real training, synthetic testing')
    ax.tick_params(axis='x',rotation=20)
    q=R[(R.protocol=='TSTR')&(R.synth=='wgangp')].groupby(['seed','fold']).accuracy.mean().reset_index()
    ax=axes[1,1]
    for i,values in enumerate([q.groupby('seed').accuracy.mean(),q.groupby('fold').accuracy.mean()]):
        ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color=[purple,blue][i],s=25)
        ax.hlines(values.mean(),i-.25,i+.25,color=ink)
    ax.set(xticks=[0,1],xticklabels=['Fold-averaged\n(by seed)','Seed-averaged\n(by fold)'],ylabel='TSTR accuracy',xlim=(-.5,1.5),ylim=(.5,1.02),title='Descriptive run variation')
    for ax,l in zip(axes.flat,'abcd'):ax.text(-.16,1.10,l,transform=ax.transAxes,weight='bold',fontsize=11)
    fig.savefig(FIGURES/'fig6_utility.pdf',bbox_inches='tight');plt.close(fig)
    methods=['wgangp','ctgan','tvae','copula','bootstrap']
    labels=['WGAN-GP','CTGAN','TVAE','Copula','Bootstrap']
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.8),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','macro_auc','mia_auc'],['TSTR accuracy','TSTR macro-AUC','Membership AUC']):
        if metric=='mia_auc':
            q=M.groupby('synth')[metric].agg(['mean','std'])
        else:
            q=R[R.protocol=='TSTR'].groupby(['synth','seed','fold'])[metric].mean().groupby('synth').agg(['mean','std'])
        v=q.loc[methods]
        ax.bar(np.arange(5),v['mean'],.65,yerr=v['std'],capsize=2,
               color=[purple,'#669bbc',blue,'#d99246','#90969d'])
        ax.set(xticks=range(5),xticklabels=labels,ylabel=title,ylim=(0,1.1))
        ax.tick_params(axis='x',rotation=38)
        ref=.5 if metric=='mia_auc' else R[R.protocol=='TRTR'].groupby('fold')[metric].mean().mean()
        ax.axhline(ref,c=ink,ls='--',lw=.9)
    for ax,l in zip(axes,'abc'):ax.text(-.18,1.08,l,transform=ax.transAxes,weight='bold',fontsize=11)
    fig.savefig(FIGURES/'fig7_baselines.pdf',bbox_inches='tight');plt.close(fig)
    print('Audited utility figures generated.')

if __name__ == '__main__':
    main()
