"""Write a complete SHA-256 manifest and create the versioned upload archive."""
import argparse
import hashlib
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo
from pathlib import Path
from common import ROOT, release_files

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT.parent/'zenodo_release.zip')
    a=p.parse_args()
    if ROOT in a.archive.resolve().parents:
        p.error('Write the archive outside the release folder')
    paths=sorted(release_files())
    (ROOT/'CHECKSUMS.txt').write_text(''.join(
        f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT)}\n' for f in paths))
    paths.append(ROOT/'CHECKSUMS.txt')
    with ZipFile(a.archive,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(paths):
            info=ZipInfo('zenodo_release/'+str(f.relative_to(ROOT)),date_time=(2026,9,15,0,0,0))
            info.compress_type=ZIP_DEFLATED
            info.external_attr=0o100644<<16
            z.writestr(info,f.read_bytes())
    print(f'{len(paths)} files archived: {a.archive}')
