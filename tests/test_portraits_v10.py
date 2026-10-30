from pathlib import Path
import json,hashlib
import pytest
ROOT=Path(__file__).resolve().parents[1]
def test_new_portrait_provenance():
    rows=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    new=[r for r in rows if r.get('collection')=='chinese_celebrities']
    assert len(new)==12 and len(rows)==33
    for r in new:
        assert r['author'] and r['source_url'].startswith('https://commons.wikimedia.org/')
        assert r['license'].startswith(('CC BY','CC0'))
        assert (ROOT/'samples'/r['source_evidence']).is_file()
        assert hashlib.sha256((ROOT/'samples'/r['filename']).read_bytes()).hexdigest()==r['sha256']
def test_sample_selection_is_manifest_only():
    from app import select_sample
    assert Path(select_sample('31_zhou_xun.jpg')).is_file()
    with pytest.raises(Exception):select_sample('../README.md')
def test_existing_sources_unchanged():
    import subprocess
    old=json.loads(subprocess.check_output(['git','show','72227dd:samples/metadata.json'],cwd=ROOT))['images']
    current=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    for r in old:
        assert next(x for x in current if x['filename']==r['filename'])==r
        original=subprocess.check_output(['git','show','72227dd:samples/'+r['filename']],cwd=ROOT)
        assert (ROOT/'samples'/r['filename']).read_bytes()==original
