from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import time
import zipfile
import threading
import cv2
import numpy as np
import pytest
from PIL import Image,ImageCms
from cartoonify.core import STYLES,PRESETS,Params,process,resize,face_mask,region_cleanup,normalized
from cartoonify.cache import ByteLRU
from cartoonify.service import job,cancel_job,batch_export,load_cached,encode

def scene(h=160,w=240):
    yy,xx=np.mgrid[:h,:w]
    result=np.stack([(xx*255/max(1,w-1)),(yy*255/max(1,h-1)),(xx+yy)%256],axis=-1).astype(np.uint8)
    cv2.circle(result,(w//2,h//2),h//4,(210,160,130),-1)
    cv2.line(result,(w//3,h//2),(w*2//3,h//2),(50,40,30),2)
    return result

@pytest.mark.parametrize('style',STYLES)
def test_determinism_and_cache(style):
    im=scene();cache=ByteLRU(20*1024**2)
    p=Params(*PRESETS[STYLES.index(style)])
    a=process(im,style,p,cache=cache);b=process(im,style,p,cache=cache)
    assert np.array_equal(a,b)
    assert cache.stats()['hits']>=2
    before=cache.stats()['misses'];process(im,style,replace(p,outline=.15),cache=cache)
    assert cache.stats()['misses']==before

def test_cache_evict_and_invalidate():
    cache=ByteLRU(900)
    for i in range(10): cache.compute(i,lambda:np.ones((10,10),np.float32))
    assert cache.used<=900 and len(cache.items)==2
    a=cache.compute('immutable',lambda:np.zeros(20,np.uint8))
    with pytest.raises(ValueError): a[0]=2
    calls=[]
    with ThreadPoolExecutor(3) as pool: list(pool.map(lambda _:cache.compute('shared',lambda:(calls.append(1),np.ones(20))[1]),range(3)))
    assert len(calls)==1

@pytest.mark.parametrize('style',STYLES)
def test_tile_seams(style):
    im=scene(180,270);p=Params(protect=False)
    tiled=process(im,style,p,tile_size=64,cache=None)
    single=process(im,style,p,tile_size=512,cache=None)
    assert np.max(np.abs(tiled.astype(np.int16)-single.astype(np.int16)))<=1

@pytest.mark.parametrize('style',STYLES)
def test_preview_export_consistency(style):
    im=scene(760,1140);p=Params(protect=False)
    preview=process(im,style,p,preview=True,quality='标准')
    full=process(im,style,p,quality='精细')
    down=resize(full,1000)
    error=np.abs(preview.astype(np.float32)-down.astype(np.float32)).mean()
    assert error<15, error

@pytest.mark.parametrize('style',[STYLES[0],STYLES[2]])
def test_original_resolution_guided_tiles(style):
    im=scene(1010,1120);p=Params(protect=False)
    a=process(im,style,p,tile_size=128)
    b=process(im,style,p,tile_size=512)
    assert np.max(np.abs(a.astype(np.int16)-b.astype(np.int16)))<=1

def test_styles_are_distinct():
    im=scene();outs=[process(im,s) for s in STYLES]
    for i in range(4):
        for j in range(i): assert np.abs(outs[i].astype(float)-outs[j].astype(float)).mean()>5

def test_manual_protection_and_invalid_box():
    im=scene();mask,boxes=face_mask(im,'[[0.3,0.2,0.4,0.6]]')
    assert boxes and mask[80,120]>.8 and mask[0,0]<.05
    p=Params(face_boxes='[[0.3,0.2,0.4,0.6]]',protect=True)
    a=process(im,STYLES[2],p);b=process(im,STYLES[2],replace(p,protect=False))
    assert not np.array_equal(a,b)
    with pytest.raises(ValueError): face_mask(im,'[[0.8,0,0.5,0.5]]')

def test_region_speckle_cleanup():
    labels=np.zeros((40,40),np.uint8);labels[:,20:]=1;labels[10,10]=1;labels[30,30]=0
    out=region_cleanup(labels,2)
    assert out[10,10]==0 and out[30,30]==1
    assert np.all(out[:,19]==0) and np.all(out[:,20]==1)

@pytest.mark.parametrize('style,field,a,b',[(STYLES[2],'colors',4,12),(STYLES[2],'dot_spacing',4,14),(STYLES[2],'dot_angle',0,45),(STYLES[2],'dot_opacity',0,.6),(STYLES[3],'pencil_shade',0,.7),(STYLES[3],'paper',0,.4),(STYLES[0],'outer',0,1),(STYLES[0],'inner',0,1)])
def test_visible_controls_change_output(style,field,a,b):
    im=scene();p=Params(protect=False)
    x=process(im,style,replace(p,**{field:a}));y=process(im,style,replace(p,**{field:b}))
    assert np.abs(x.astype(float)-y.astype(float)).mean()>.05

def test_reject_nonfinite():
    for field in ('outline','contrast','smooth','dot_spacing'):
        with pytest.raises(ValueError): normalized(replace(Params(),**{field:float('nan')}),STYLES[0])

def test_cancellation_and_resource_serialization():
    active=[0,0];lock=threading.Lock()
    def work(i):
        with job(str(i)):
            with lock: active[0]+=1;active[1]=max(active)
            time.sleep(.03)
            with lock: active[0]-=1
    with ThreadPoolExecutor(3) as pool: list(pool.map(work,range(3)))
    assert active[1]==1
    with job('cancel-test') as token:
        cancel_job('cancel-test')
        with pytest.raises(InterruptedError): process(scene(),STYLES[0],cancel=token.is_set)

def test_formats_icc_and_decode_invalidation(tmp_path):
    from cartoonify.core import load_image
    p=tmp_path/'test.webp';Image.fromarray(scene()).save(p,lossless=True)
    assert load_image(p).shape==(160,240,3)
    p=tmp_path/'profile.png';profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(scene()).save(p,icc_profile=profile)
    assert np.max(np.abs(load_image(p).astype(int)-scene().astype(int)))<=1
    first=load_cached(p).copy();Image.new('RGB',(30,40),'red').save(p)
    assert load_cached(p).shape!=(first.shape)
    for fmt in ('PNG','JPG'):
        from io import BytesIO
        out=Image.open(BytesIO(encode(scene(),fmt)))
        assert out.info.get('icc_profile')

def test_partial_batch_failure(tmp_path):
    good=tmp_path/'ok.png';bad=tmp_path/'bad.png'
    Image.fromarray(scene()).save(good);bad.write_bytes(b'bad')
    archive,status=batch_export([good,bad],STYLES[0],Params(),session='test-batch')
    with zipfile.ZipFile(archive) as z:
        assert len([n for n in z.namelist() if n.endswith('.png')])==1
        assert 'bad.png' in z.read('处理记录.txt').decode()
    Path(archive).unlink()

def test_ui_queue_and_style_controls():
    from app import build_app,defaults
    demo=build_app()
    heavy=[f for f in demo.fns.values() if f.api_name in ('preview','export','batch')]
    assert len(heavy)==3 and {f.concurrency_id for f in heavy}=={'cpu'}
    assert defaults(STYLES[2])[2]['maximum']==12
    assert defaults(STYLES[0])[2]['visible'] is False
