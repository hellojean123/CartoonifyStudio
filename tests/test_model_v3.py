"""Verify distributed model and the previously missed close crop."""
from pathlib import Path
import hashlib,json
from cartoonify.core import face_mask,load_image,resize

def test_model_matches_source_manifest():
    root=Path(__file__).parents[1]/'models'
    info=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root/info['name']).read_bytes()).hexdigest()==info['sha256']
    assert 'MIT License' in (root/'LICENSE-YuNet.txt').read_text(encoding='utf-8')

def test_previously_missed_close_crop_has_landmarks():
    image=resize(load_image(Path(__file__).parents[1]/'samples/01_portrait.jpg'),1000)
    mask,boxes,landmarks=face_mask(image,details=True)
    assert boxes and landmarks
    assert mask.max()>.9
    assert landmarks[0][1].shape==(5,2)
