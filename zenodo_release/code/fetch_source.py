"""Download the four source files from the fixed figshare version, checking MD5."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from common import RESULTS
API = 'https://api.figshare.com/v2/articles/23522490/versions/6'
NAMES = ['Asthma_peaktable_ver3.csv', 'Bronchi_peaktable_ver3.csv',
         'COPD_peaktable_ver3.csv', 'intersection_of_detected_compunds.xlsx']
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=RESULTS / 'source')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        metadata = json.load(response)
    files = {f['name']: f for f in metadata['files']}
    for name in NAMES:
        f = files[name]
        with urllib.request.urlopen(f['download_url'], timeout=60) as response:
            content = response.read()
        expected = f.get('computed_md5') or f.get('supplied_md5')
        if expected and hashlib.md5(content).hexdigest() != expected:
            raise ValueError(f'Source checksum mismatch: {name}')
        (a.output_dir / name).write_bytes(content)
        print(name, len(content))
    (a.output_dir / 'figshare_v6.json').write_text(json.dumps(metadata, indent=2) + '\n')
