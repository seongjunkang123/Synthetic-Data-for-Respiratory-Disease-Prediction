"""Embed generated numerical inputs into a single manuscript text source."""
import argparse
import re
from pathlib import Path

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manuscript-dir',type=Path,required=True)
    a=p.parse_args()
    original=a.manuscript_dir/'breathnet_data_descriptor.tex'
    text=original.read_text()
    def embed(m):
        path=a.manuscript_dir/(m[1]+('' if m[1].endswith('.tex') else '.tex'))
        content=path.read_text()
        assert 'pending' not in content.lower(),f'Unfinished numerical input: {path}'
        return '% Embedded from '+m[1]+'\n'+content
    text=re.sub(r'\\input\{([^}]+)\}',embed,text)
    assert r'\input{' not in text and r'\bibliography{' not in text
    out=a.manuscript_dir/'breathnet_submission.tex'
    out.write_text(text)
    print(out)
