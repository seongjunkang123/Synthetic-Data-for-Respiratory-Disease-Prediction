"""Verify the file manifest, data integrity, benchmark provenance and metric totals."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from common import ROOT, DATA, RESULTS, CLASSES, Preprocessor, load_real, validate_sample, release_files
import cv_splits as S

def check_data():
    real, X, y, cols = load_real()
    syn = pd.read_csv(DATA/'synthetic_release_6000.csv')
    ys = pd.Categorical(syn.Disease,categories=CLASSES).codes.astype(int)
    validate_sample(syn[cols].to_numpy(),ys,2000)
    assert list(syn.columns) == list(real.columns)
    assert syn['Patient ID'].is_unique and syn['Patient ID'].str.startswith('syn_').all()
    parts = pd.concat([pd.read_csv(DATA/f'synthetic_{c}.csv') for c in CLASSES],ignore_index=True)
    pd.testing.assert_frame_equal(parts,syn)
    dictionary = pd.read_csv(DATA/'voc_dictionary.csv')
    assert dictionary.shape == (78,2)
    assert dictionary.pubchem_cid.astype(str).tolist() == cols
    pre=Preprocessor.load(ROOT/'model/preprocessing.json')
    np.testing.assert_allclose(pre.scaler.data_min_,np.log1p(X).min(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(pre.scaler.data_max_,np.log1p(X).max(0),rtol=1e-12,atol=1e-12)
    Z=pre.transform(syn[cols].to_numpy())
    assert Z.min()>=-1e-7 and Z.max()<=1+1e-7
    return real,X,y,cols

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-checksums',action='store_true')
    p.add_argument('--data-only',action='store_true')
    a=p.parse_args()
    real,X,y,cols=check_data()
    checks={'profiles':6000,'features':78,'class_counts':[2000]*3,'data_integrity':'pass'}
    if not a.skip_checksums:
        manifest=ROOT/'CHECKSUMS.txt'
        expected=set()
        for line in manifest.read_text().splitlines():
            sha,name=line.split(maxsplit=1)
            path=ROOT/name
            assert path.is_file(),f'Missing manifest entry: {name}'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,f'Checksum mismatch: {name}'
            expected.add(name)
        actual={str(f.relative_to(ROOT)) for f in release_files()}
        assert actual==expected, f'Manifest coverage differs: {actual ^ expected}'
        checks['manifest_files']=len(expected)
    if not a.data_only:
        root=RESULTS/'audited'
        complete=json.loads((root/'COMPLETE.json').read_text())
        assert set(complete['methods'])=={'wgangp','ctgan','tvae','copula','bootstrap'}
        assert complete['seeds']==[0,1,2] and complete['folds']==5
        for k,(tr,te) in enumerate(S.get_folds()):
            assert not set(tr)&set(te) and sorted(tr+te)==list(range(121))
            for method in complete['methods']:
                for seed in ([0] if method=='copula' else range(3)):
                    path=root/'synth'/f'{method}_f{k}_s{seed}.npz'
                    sample=np.load(path)
                    validate_sample(sample['X'],sample['y'],500)
                    meta=json.loads(path.with_suffix('.json').read_text())
                    assert meta['train_indices']==tr and meta['test_indices']==te
                    assert meta['columns']==cols
                    if method=='wgangp':
                        pre=Preprocessor.load(path.with_name(path.stem+'_preprocessing.json'))
                        expected=Preprocessor().fit(X[tr])
                        np.testing.assert_allclose(pre.scaler.data_min_,expected.scaler.data_min_,atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(pre.scaler.data_max_,expected.scaler.data_max_,atol=1e-12,rtol=1e-12)
                        scaled=pre.transform(sample['X'])
                        # Float32 inverse/forward transforms can overshoot a boundary by a few ULPs.
                        assert scaled.min()>=-1e-6 and scaled.max()<=1+1e-6
        scores=pd.read_csv(root/'cv_results2.csv')
        assert not scores[['accuracy','macro_f1','macro_auc']].isna().any().any()
        assert len(scores[scores.protocol=='TRTR'])==20
        assert len(scores[scores.protocol=='TSTR'])==260
        assert len(scores[scores.protocol=='TRTS'])==260
        predictions=pd.read_csv(root/'cv_predictions2.csv')
        for key,sub in predictions.groupby(['fold','synth','seed','protocol','clf']):
            match=scores
            for col,value in zip(['fold','synth','seed','protocol','clf'],key):
                match=match[match[col]==value]
            assert len(match)==1
            assert sub.record_index.is_unique
            assert set(sub.true) == {0,1,2}
            probs=sub[['p0','p1','p2']].to_numpy()
            assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
            np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
            assert np.isclose((sub.true==sub.pred).mean(),match.accuracy.iloc[0])
            assert np.isclose(f1_score(sub.true,sub.pred,average='macro'),match.macro_f1.iloc[0])
            assert np.isclose(roc_auc_score(sub.true,probs,multi_class='ovr',average='macro'),match.macro_auc.iloc[0])
            if key[3] in ['TRTR','TSTR']:
                assert set(sub.record_index)==set(S.get_folds()[key[0]][1])
                assert np.array_equal(sub.true.to_numpy(),y[sub.record_index.to_numpy()])
            else:
                assert key[3]=='TRTS' and len(sub)==1500
                sample=np.load(root/'synth'/f'{key[1]}_f{key[0]}_s{key[2]}.npz')
                assert set(sub.record_index)==set(range(1500))
                assert np.array_equal(sub.true.to_numpy(),sample['y'][sub.record_index.to_numpy()])
        assert len(predictions.groupby(['fold','synth','seed','protocol','clf'])) == len(scores)
        attacks=pd.read_csv(root/'cv_mia2.csv')
        assert len(attacks)==65 and attacks.mia_auc.between(0,1).all()
        fidelity=pd.read_csv(root/'heldout_fidelity.csv')
        assert len(fidelity)==195
        assert fidelity[['heldout_ks','training_ks']].apply(lambda c:c.between(0,1).all()).all()
        checks.update(benchmark_samples=65,evaluation_rows=len(scores),predictions=len(predictions))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
