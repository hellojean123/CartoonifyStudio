"""Layer semantics and rendering regressions, not automated aesthetic scores."""
from dataclasses import replace
import cv2
import numpy as np
from cartoonify.core import Params,STYLES,process,resize
from cartoonify.cache import ByteLRU
from cartoonify.pencil import graphite_strokes


def test_smooth_shadows_have_tone_without_photo_texture():
    ramp=np.tile(np.linspace(40,245,240,dtype=np.uint8),(160,1))
    im=np.repeat(ramp[...,None],3,axis=2)
    p=Params(outline=0,paper=0,protect=False)
    out=process(im,STYLES[3],p,cache=None)
    assert out[:,20:60].mean()<out[:,120:160].mean()-20
    assert out[:,-20:].mean()>250
    assert out[:,20:60].std()>1  # graphite is present on smooth, shaded forms


def test_short_strokes_are_deterministic_even_on_uniform_dark_regions():
    tone=np.full((140,180),.25,np.float32)
    angle=np.zeros_like(tone);coherence=np.zeros_like(tone)
    a=graphite_strokes(tone,angle,coherence)
    b=graphite_strokes(tone,angle,coherence)
    assert np.array_equal(a,b) and np.isfinite(a).all()
    assert a.std()>.05
    assert not np.allclose(a,np.roll(a,4,axis=1))


def test_colored_pencil_deposits_pigment_on_white_highlights():
    im=np.full((120,180,3),255,np.uint8);im[20:100,30:150]=(180,45,35)
    p=Params(outline=0,protect=False,paper=0,colored=True)
    out=process(im,STYLES[3],p,cache=None)
    assert out[:10,:10].min()==255
    center=out[35:85,50:130].mean(axis=(0,1))
    assert center[0]>center[1]+20 and center[0]>center[2]+20


def test_render_controls_reuse_structure_and_tone_layers():
    yy,xx=np.mgrid[:120,:180]
    im=np.repeat(((xx+yy)%240).astype(np.uint8)[...,None],3,axis=2)
    cache=ByteLRU(10*1024**2);p=Params(protect=False)
    first=process(im,STYLES[3],p,cache=cache)
    misses=cache.stats()['misses']
    for field,value in [('colored',True),('pencil_shade',.7),('paper',.4),('outline',.1)]:
        result=process(im,STYLES[3],replace(p,**{field:value}),cache=cache)
        assert cache.stats()['misses']==misses
        assert not np.array_equal(first,result)


def test_high_resolution_paper_has_no_tile_seams_and_matches_preview():
    yy,xx=np.mgrid[:1080,:1440]
    gray=(100+50*np.cos(xx/160)*np.sin(yy/190)).astype(np.uint8)
    image=np.repeat(gray[...,None],3,axis=2);p=Params(protect=False,paper=.4)
    a=process(image,STYLES[3],p,tile_size=127)
    b=process(image,STYLES[3],p,tile_size=512)
    assert np.max(np.abs(a.astype(int)-b.astype(int)))<=1
    preview=process(image,STYLES[3],p,preview=True)
    assert np.abs(preview.astype(float)-resize(b,1000).astype(float)).mean()<5


def test_bright_face_keeps_broad_shadows_without_a_dark_outline():
    yy,xx=np.mgrid[:200,:200]
    luminance=205-25*np.exp(-((xx-85)**2+(yy-110)**2)/1200)
    im=np.stack([luminance+20,luminance,luminance-15],axis=-1).clip(0,255).astype(np.uint8)
    p=Params(outline=0,paper=0,face_boxes='[[0.1,0.05,0.8,0.9]]')
    out=process(im,STYLES[3],p,cache=None)
    assert out[95:125,70:100].mean()<out[40:65,120:145].mean()-5
