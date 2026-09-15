"""Rebuild the shared-CID matrix from the version 6 figshare source files."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import DATA, RESULTS, CLASSES
FILES = ['Asthma_peaktable_ver3.csv', 'Bronchi_peaktable_ver3.csv', 'COPD_peaktable_ver3.csv']

def build(src, duplicate_policy='first'):
    inter = pd.read_excel(src / 'intersection_of_detected_compunds.xlsx', index_col='pubchem_CID')
    tables = [pd.read_csv(src / f, index_col='pubchem_CID') for f in FILES]
    keep = sorted(set.intersection(*(set(t.index) for t in tables)))
    assert keep == sorted(inter.index.tolist()) and len(keep) == 78
    names = inter['IUPAC Name'].to_dict()
    rows, audit = [], []
    for label, tab in zip(CLASSES, tables):
        patcols = [c for c in tab.columns if c != 'IUPAC Name']
        for cid in tab.index[tab.index.duplicated(keep=False)].unique():
            audit.append(dict(disease=label, cid=int(cid), occurrences=int((tab.index == cid).sum()),
                              included=bool(cid in keep), policy=duplicate_policy))
        numeric = tab[patcols].apply(pd.to_numeric)
        if duplicate_policy == 'mean':
            numeric = numeric.groupby(level=0, sort=False).mean()
        else:
            numeric = numeric[~numeric.index.duplicated(keep=duplicate_policy)]
        numeric = numeric.loc[keep]
        assert np.isfinite(numeric.to_numpy()).all()
        for j, col in enumerate(patcols):
            rows.append([f'{label}_{j+1}', label] + numeric[col].tolist())
    df = pd.DataFrame(rows, columns=['Patient ID', 'Disease'] + [str(c) for c in keep])
    voc = pd.Series({c: names[c] for c in keep}, name='iupac_name').rename_axis('pubchem_cid')
    return df, voc, audit

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'rebuilt')
    p.add_argument('--duplicate-policy', choices=['first', 'last', 'mean'], default='first')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    df, voc, audit = build(a.source_dir, a.duplicate_policy)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output_dir / 'real_121x78.csv', index=False)
    voc.to_csv(a.output_dir / 'voc_dictionary.csv')
    (a.output_dir / 'duplicate_resolution.json').write_text(json.dumps(audit, indent=2) + '\n')
    if a.verify:
        ref = pd.read_csv(DATA / 'real_121x78.csv')
        assert df.columns.equals(ref.columns) and df.iloc[:, :2].equals(ref.iloc[:, :2])
        np.testing.assert_allclose(df.iloc[:, 2:], ref.iloc[:, 2:], rtol=1e-12, atol=1e-12)
        print('Rebuilt values and labels match the archived real matrix (rtol=atol=1e-12).')
    print(df.shape, json.dumps(audit))

if __name__ == '__main__':
    main()
