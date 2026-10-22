import numpy as np
import pytest
from PIL import Image
from cartoonify.core import STYLES, Params, process, load_image

@pytest.mark.parametrize('style', STYLES)
@pytest.mark.parametrize('shape', [(1,1,3),(32,47,3),(240,320,3)])
def test_shapes(style, shape):
    im = np.random.default_rng(4).integers(0,256,shape,dtype=np.uint8)
    out = process(im, style)
    assert out.shape==im.shape and out.dtype==np.uint8

@pytest.mark.parametrize('style', STYLES)
def test_extremes(style):
    im = np.full((100,100,3), 128, np.uint8)
    for p in [Params(-99,-99,-99,-99), Params(99,99,99,99,True)]:
        assert process(im,style,p).shape==im.shape

def test_loading(tmp_path):
    p=tmp_path/'alpha.png'
    Image.new('RGBA',(10,20),(255,0,0,0)).save(p)
    assert np.all(load_image(p)==255)
    Image.new('L',(10,20),100).save(p)
    assert load_image(p).shape==(20,10,3)
    Image.new('RGB',(5000,4000)).save(p)
    with pytest.raises(ValueError): load_image(p)
    p.write_bytes(b'not an image')
    with pytest.raises(Exception): load_image(p)

def test_exif(tmp_path):
    p=tmp_path/'rotated.jpg'
    im=Image.new('RGB',(20,10)); exif=im.getexif(); exif[274]=6
    im.save(p,exif=exif)
    assert load_image(p).shape==(20,10,3)

def test_validation():
    with pytest.raises(ValueError): process(np.zeros((10,10,3),np.uint8),'bad')
    with pytest.raises(ValueError): process(np.zeros((0,10,3),np.uint8),STYLES[0])
    assert process(np.zeros((20,20),np.uint8),STYLES[0]).shape==(20,20,3)
