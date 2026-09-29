"""Restore the exact model assets pinned in models/manifest.json."""
from pathlib import Path
import hashlib,json,urllib.request,shutil
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'models/manifest.json').read_text())
for role,model in manifest.items():
    for entry in model['files']:
        target=root/'models'/role/entry['file']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        url=entry['download_url']
        temp=target.with_suffix(target.suffix+'.download')
        with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as output:shutil.copyfileobj(response,output)
        if hashlib.sha256(temp.read_bytes()).hexdigest()!=entry['sha256']:raise RuntimeError('Model asset hash mismatch: '+entry['file'])
        temp.replace(target)
