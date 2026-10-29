"""CPU portrait integration regressions; visual quality is reviewed separately."""
from dataclasses import replace
import json,hashlib
from pathlib import Path
import cv2,numpy as np,pytest
from cartoonify import anime
from cartoonify.core import Params,STYLES,normalized,process
from cartoonify.cache import ByteLRU


def test_webtoon_integrity_and_exclusive_cpu_provider():
    root=Path(__file__).resolve().parents[1]/'models'
    record=json.loads((root/'anime_webtoon_metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root/'webtoon.onnx').read_bytes()).hexdigest()==record['onnx_sha256']
    assert record['max_abs_error']<.002 and max(record['mean_errors'])<.00005
    assert record['cuda'] is None
    first=anime.session('webtoon');assert first.get_providers()==['CPUExecutionProvider']
    assert anime.session('webtoon') is first
    second=anime.session('v2');assert second is not first
    assert anime._SESSION is second and anime._SESSION_NAME=='v2'


def test_eye_alignment_does_not_change_face_proportions():
    box=(50,40,140,180);points=np.float32([[90,100],[150,120],[120,145],[100,180],[145,185]])
    m=anime.head_transform(box,points,image_shape=(400,400,3))
    mapped=cv2.transform(points[None],m)[0]
    assert abs(mapped[0,1]-mapped[1,1])<.0001
    a=np.linalg.norm(mapped[0]-mapped[1])/np.linalg.norm(points[0]-points[1])
    b=np.linalg.norm(mapped[2]-mapped[3])/np.linalg.norm(points[2]-points[3])
    assert abs(a-b)<.0001
    for bad in ((0,0,10,20),(0,0,float('nan'),100),(0,0,1e15,100),(0,0,400,80)):
        assert anime.head_transform(bad) is None


def test_head_crop_at_image_edge_has_no_reflected_face(monkeypatch):
    captured=[]
    def infer(im,name='v2'):
        captured.append(im.copy());return im.copy()
    monkeypatch.setattr(anime,'_infer',infer)
    im=np.zeros((120,180,3),np.uint8);im[:]=np.arange(180,dtype=np.uint8)[None,:,None]
    result=anime.stylize(im,[(0,0,100,110)])
    assert result.shape==im.shape and len(captured)==2
    assert captured[1].shape[:2]==(256,256)
    # Replication/clamped crops never reverse the image's left-to-right gradient.
    assert np.diff(captured[1][128,:,0].astype(int)).min()>=0
    alpha=anime._head_alpha()
    assert np.max(alpha[[0,-1]])==0 and np.max(alpha[:,[0,-1]])==0


def test_mode_changes_recompute_model_but_line_slider_reuses_it(monkeypatch):
    calls=[]
    def stylize(im,*args,**kwargs):
        calls.append(kwargs['mode']);return np.full_like(im,80 if kwargs['mode']=='动漫增强' else 180)
    monkeypatch.setattr(anime,'stylize',stylize)
    im=np.full((60,80,3),140,np.uint8);cache=ByteLRU(20*1024**2);p=Params(protect=False)
    a=process(im,STYLES[0],p,cache=cache)
    process(im,STYLES[0],replace(p,outline=.1),cache=cache)
    b=process(im,STYLES[0],replace(p,anime_mode='柔和插画'),cache=cache)
    assert calls==['动漫增强','柔和插画'] and b.mean()>a.mean()+80
    with pytest.raises(ValueError):normalized(replace(p,anime_mode='unknown'),STYLES[0])


def test_skin_tone_constraint_does_not_copy_source_texture():
    yy,xx=np.mgrid[:180,:180]
    source=np.full((180,180,3),(140,95,65),np.uint8)
    source[::2,::2]=[150,100,70]
    generated=np.full_like(source,(220,180,155));generated[80:90,70:110]=245
    result=anime.preserve_skin_tone(source,generated,[(20,10,140,160)])
    assert result[100,90,0]<generated[100,90,0]-15
    assert np.std(result[100:110,80:100].astype(float),axis=(0,1)).max()<2
    assert result[85,90].mean()>result[100,90].mean()+30


def test_cancel_between_global_and_head_pass(monkeypatch):
    calls=[]
    def infer(im,name='v2'):
        calls.append(name);return im
    monkeypatch.setattr(anime,'_infer',infer)
    with pytest.raises(InterruptedError):
        anime.stylize(np.zeros((200,200,3),np.uint8),[(30,30,100,120)],cancel=lambda:len(calls)>0)
    assert len(calls)==1


def test_ui_mode_reset_and_shared_queue():
    from app import build_app,defaults
    from dataclasses import fields
    count=len(fields(Params));values=defaults(STYLES[0])
    index=[f.name for f in fields(Params)].index('anime_mode')
    assert len(values)==count+2 and values[index]['value']=='动漫增强'
    assert defaults(STYLES[3])[index]['visible'] is False
    app=build_app()
    assert any(getattr(b,'label',None)=='动漫表现' for b in app.blocks.values())
    assert all(f.concurrency_id=='cpu' for f in app.fns.values() if f.api_name in ('preview','export','batch'))


def test_head_refinement_ignores_cascade_only_and_tiny_model_faces(monkeypatch):
    from cartoonify.core import _layers
    seen=[]
    def stylize(im,boxes,*args,**kwargs):
        seen.append(list(boxes));return im.copy()
    monkeypatch.setattr(anime,'stylize',stylize)
    im=np.full((180,200,3),150,np.uint8);mask=np.zeros(im.shape[:2],np.float32)
    prepared={'guide':im,'blur':im,'mask':mask,'boxes':[(20,20,90,100)],'landmarks':[(np.array([10,10,53,58]),np.zeros((5,2)))]}
    _layers(prepared,STYLES[0],Params())
    assert seen==[[]]
    _layers(prepared,STYLES[0],Params(face_boxes='[[0.1,0.1,0.5,0.6]]'))
    assert seen[-1]==[(20,18,100,108)]
    _layers(prepared,STYLES[0],Params(protect=False,face_boxes='invalid ignored when protection is off'))
    assert seen[-1]==[]


def test_bright_low_contrast_head_preconditioning_is_selective():
    yy,xx=np.mgrid[:512,:512]
    gray=(195+40*xx/512).astype(np.uint8)
    crop=np.repeat(gray[...,None],3,axis=2)
    cv2.line(crop,(180,220),(230,220),(180,180,180),4)
    output,enhanced=anime.prepare_head_input(crop)
    assert enhanced and output.shape==crop.shape
    assert np.abs(output.astype(float)-crop.astype(float)).mean()>1
    normal=np.clip(crop.astype(int)-90,0,255).astype(np.uint8)
    same,enhanced=anime.prepare_head_input(normal)
    assert not enhanced and same is normal
    flat=np.full_like(crop,220)
    same,enhanced=anime.prepare_head_input(flat)
    assert not enhanced and np.array_equal(same,flat)
