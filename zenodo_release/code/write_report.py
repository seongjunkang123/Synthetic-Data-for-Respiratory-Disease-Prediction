"""Generate manuscript numbers and tables directly from complete audited results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, RESULTS
LABELS = {'logreg':'Logistic regression','rf':'Random forest','xgb':'XGBoost','mlp':'Multilayer perceptron'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draft',action='store_true',help='Create explicit pending placeholders for layout checks')
    a=p.parse_args()
    out=RESULTS/'report';out.mkdir(exist_ok=True)
    macro_names=['TstrAcc','TstrSd','RealAcc','RealSd','BestClassifier','BestAcc',
                 'RecallAsthma','RecallBronchi','RecallCopd','CtganAcc','TvaeAcc',
                 'CopulaAcc','BootstrapAcc','ExampleReal','ExampleBootstrap',
                 'ExampleSynthetic','ExampleAugmented','MiaMean','MiaSd',
                 'HeldoutAsthma','HeldoutBronchi','HeldoutCopd','TrainingAsthma','TrainingBronchi','TrainingCopd']
    if a.draft:
        macros={n:r'\textit{pending}' for n in macro_names}
        utility=(r'\begin{table}[htbp]\centering\caption{\textbf{Held-out utility.} Audited evaluation in progress.}'
                 r'\label{tab:utility}\begin{tabular}{l}Pending complete benchmark results\end{tabular}\end{table}')
    else:
        root=RESULTS/'audited'
        assert (root/'COMPLETE.json').exists(),'Benchmark is not complete'
        R=pd.read_csv(root/'cv_results2.csv');P=pd.read_csv(root/'cv_predictions2.csv')
        M=pd.read_csv(root/'cv_mia2.csv')
        E=pd.read_csv(RESULTS/'example/benchmark_runs.csv')
        run=R[R.protocol=='TSTR'].groupby(['synth','fold','seed']).accuracy.mean()
        real=R[R.protocol=='TRTR'].groupby('fold').accuracy.mean()
        w=run.loc['wgangp']
        best=R[(R.protocol=='TSTR')&(R.synth=='wgangp')].groupby('clf').accuracy.mean().idxmax()
        rec=[(q.true==q.pred).mean() for _,q in P[(P.protocol=='TSTR')&(P.synth=='wgangp')].groupby('true')]
        mia=M[M.synth=='wgangp'].mia_auc
        ev=E.groupby('training').accuracy.mean()
        values=dict(TstrAcc=w.mean(),TstrSd=w.std(),RealAcc=real.mean(),RealSd=real.std(),
            BestClassifier=LABELS[best],BestAcc=R[(R.protocol=='TSTR')&(R.synth=='wgangp')&(R.clf==best)].accuracy.mean(),
            RecallAsthma=rec[0],RecallBronchi=rec[1],RecallCopd=rec[2],
            CtganAcc=run.loc['ctgan'].mean(),TvaeAcc=run.loc['tvae'].mean(),
            CopulaAcc=run.loc['copula'].mean(),BootstrapAcc=run.loc['bootstrap'].mean(),
            ExampleReal=ev['real_only'],ExampleBootstrap=ev['bootstrap_only'],
            ExampleSynthetic=ev['synthetic_only'],ExampleAugmented=ev['real_plus_synthetic'],
            MiaMean=mia.mean(),MiaSd=mia.std())
        F=pd.read_csv(root/'heldout_fidelity.csv')
        fq=F[F.synth=='wgangp'].groupby('disease')[['heldout_ks','training_ks']].mean()
        for c,suffix in [('asthma','Asthma'),('bronchiectasis','Bronchi'),('copd','Copd')]:
            values['Heldout'+suffix]=fq.loc[c,'heldout_ks']
            values['Training'+suffix]=fq.loc[c,'training_ks']
        macros={k:v if isinstance(v,str) else f'{v:.3f}' for k,v in values.items()}
        def cell(q,col):
            return f'{q[col].mean():.3f} ({q[col].std():.3f})'
        utility=r'''\begin{table}[htbp]
\centering\footnotesize
\caption{\textbf{Held-out utility of the WGAN-GP synthesis procedure.}
Mean (standard deviation). TSTR and TRTS summarise fifteen fold/seed
combinations; TRTR summarises five real-data folds. AUC is macro-averaged
one-versus-rest. The real-data result is a reference, not an upper bound.}
\label{tab:utility}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{TSTR} & \multicolumn{2}{c}{TRTR} & \multicolumn{2}{c}{TRTS}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Classifier & Accuracy & AUC & Accuracy & AUC & Accuracy & AUC\\
\midrule
'''
        for clf in LABELS:
            cs=[]
            for proto,method in [('TSTR','wgangp'),('TRTR','real'),('TRTS','wgangp')]:
                q=R[(R.protocol==proto)&(R.synth==method)&(R.clf==clf)]
                cs += [cell(q,'accuracy'),cell(q,'macro_auc')]
            label='MLP' if clf=='mlp' else LABELS[clf]
            utility += label+' & '+' & '.join(cs)+r'\\'+'\n'
        utility+=r'\bottomrule\end{tabular}\end{table}'+'\n'
        summary=R[R.protocol=='TSTR'].groupby(['synth','fold','seed'])[['accuracy','macro_f1','macro_auc']].mean().groupby('synth').agg(['mean','std'])
        summary.to_csv(out/'method_summary.csv')
        summary_md='# Audited benchmark summary\n\n'
        summary_md+='Metrics are averages across four classifiers within each fold/seed run. Values are mean ± standard deviation over 15 runs per stochastic method and five copula folds.\n\n'
        summary_md+='| Method | TSTR accuracy | TSTR macro-AUC | Membership AUC |\n|---|---:|---:|---:|\n'
        for method in ['wgangp','ctgan','tvae','copula','bootstrap']:
            q=summary.loc[method];mi=M[M.synth==method].mia_auc
            summary_md+=f'| {method} | {q[("accuracy","mean")]:.3f} ± {q[("accuracy","std")]:.3f} | {q[("macro_auc","mean")]:.3f} ± {q[("macro_auc","std")]:.3f} | {mi.mean():.3f} ± {mi.std():.3f} |\n'
        summary_md+=f'\nReal-only reference accuracy: {real.mean():.3f} ± {real.std():.3f} over five folds.\n\n'
        summary_md+='## Runnable logistic-regression example\n\n| Training condition | Accuracy | Runs |\n|---|---:|---:|\n'
        for name,q in E.groupby('training'):
            summary_md+=f'| {name} | {q.accuracy.mean():.3f} ± {q.accuracy.std():.3f} | {len(q)} |\n'
        summary_md+='\nThe bootstrap resamples real training records and does not generate new continuous profiles. All results describe this cohort and these configurations; variation across repeated folds is not an independent-cohort confidence interval.\n'
        (ROOT/'BENCHMARK_SUMMARY.md').write_text(summary_md)
        (out/'manuscript_numbers.json').write_text(json.dumps(values,indent=2)+'\n')
    (out/'metrics.tex').write_text('% Generated by code/write_report.py\n'+''.join(
        '\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    (out/'utility.tex').write_text(utility)
    N=pd.read_csv(RESULTS/'noise_floor.csv')
    reference=r'''\begin{table}[htbp]
\centering\small
\caption{\textbf{Within-class descriptive resampling reference.}
Mean with 2.5th--97.5th percentile resampling range in brackets.
KS and Wasserstein comparisons use equally sized disjoint real halves
or a real half and synthetic draw. Correlations use class-size bootstrap
resamples. These ranges describe this training cohort, not equivalence
or population confidence intervals.}
\label{tab:reference}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcc}
\toprule
Measure & Class & Real--real & Synthetic--real\\
\midrule
'''
    for metric,title in [('ks_median',r'Median KS $D$'),('w1_median_sd',r'Median $W_1$ (real SD)'),('corr_mean_abs_diff',r'Mean $|\Delta\rho|$')]:
        for i,c in enumerate(['asthma','bronchiectasis','copd']):
            row=N[(N.metric==metric)&(N.group==c)].iloc[0]
            label='COPD' if c=='copd' else c.capitalize()
            reference+=(title if i==0 else '')+' & '+label+' & '+f'{row.real_vs_real:.3f} [{row.rvr_lo:.3f}--{row.rvr_hi:.3f}] & {row.syn_vs_real:.3f} [{row.svr_lo:.3f}--{row.svr_hi:.3f}]'+r'\\'+'\n'
        if metric!='corr_mean_abs_diff':reference+=r'\midrule'+'\n'
    reference+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    (out/'reference.tex').write_text(reference)
    print('Report tables generated',out)

if __name__=='__main__':
    main()
