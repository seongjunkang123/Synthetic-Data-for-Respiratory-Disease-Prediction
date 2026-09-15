"""Check Data Descriptor structure, references, word budgets and compiled PDF text."""
import argparse
import json
import re
from pathlib import Path
from pypdf import PdfReader

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tex',type=Path,required=True)
    p.add_argument('--allow-pending-accession',action='store_true')
    a=p.parse_args()
    text=a.tex.read_text()
    def words(start,end):
        t=text.split(r'\section*{'+start+'}',1)[1].split(r'\section*{'+end+'}',1)[0]
        t=re.sub(r'\\cite\{[^}]+\}','',t)
        return len(t.split())
    title=re.search(r'\\bfseries ([^}]+)',text)[1]
    stats=dict(title_characters=len(title),abstract_words=words('Abstract',r'Background \& Summary'),
               background_words=words(r'Background \& Summary','Methods'),
               figures=text.count(r'\begin{figure}'))
    assert stats['title_characters']<=110
    assert stats['abstract_words']<=170
    assert stats['background_words']<=700
    assert stats['figures']<=8
    for heading in ['Abstract',r'Background \& Summary','Methods','Data Records',
                    'Technical Validation','Usage Notes','Data Availability','Code Availability','Funding']:
        assert r'\section*{'+heading+'}' in text,heading
    keys=set(re.findall(r'\\bibitem\{([^}]+)\}',text))
    cited={k for match in re.findall(r'\\cite\{([^}]+)\}',text) for k in match.split(',')}
    assert cited<=keys,cited-keys
    stats['uncited_references']=sorted(keys-cited)
    if not a.allow_pending_accession:
        assert 'accession pending' not in text.lower() and 'accession will be added' not in text.lower()
    pdf=a.tex.with_suffix('.pdf')
    pages=PdfReader(pdf).pages
    rendered='\n'.join(page.extract_text() or '' for page in pages)
    assert '??' not in rendered,'Unresolved reference in PDF'
    assert 'pending' not in (a.tex.parent/'tables/metrics.tex').read_text(),'Numerical placeholders remain'
    stats['pdf_pages']=len(pages)
    stats['pending_deposition']=bool('pending' in rendered.lower())
    print(json.dumps(stats,indent=2))
