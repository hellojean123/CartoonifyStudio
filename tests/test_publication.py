from pathlib import Path
import numpy as np
import pytest
from benchmarks import legacy_v1
from cartoonify.core import STYLES

@pytest.mark.parametrize('style',STYLES)
def test_legacy_reference_is_self_contained(style):
    image=np.random.default_rng(17).integers(0,256,(72,96,3),dtype=np.uint8)
    out=legacy_v1.process(image,style)
    assert out.shape==image.shape and out.dtype==np.uint8

def test_published_evaluation_has_no_backup_dependency():
    root=Path(__file__).parents[1]
    for filename in ['evaluate.py','benchmark_v2.py','deploy_upgrade.ps1']:
        source=(root/filename).read_text(encoding='utf-8')
        assert 'backups/pre_upgrade' not in source
        assert 'backups\\pre_upgrade' not in source
