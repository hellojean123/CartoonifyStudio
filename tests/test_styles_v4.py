"""Style semantics and operational regression checks, not an aesthetic score."""
from pathlib import Path
import hashlib,json
from dataclasses import replace
import cv2,numpy as np
from cartoonify.core import process,STYLES,Params
from cartoonify.anime import session
from cartoonify.npr import flow_lines
from cartoonify.pencil import graphite_strokes

def test_bundled_anime_integrity_and_cpu():
    root=Path(__file__).resolve().parents[1]/'models'
    metadata=json.loads((root/'anime_metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((root/'face_paint_v2.onnx').read_bytes()).hexdigest()==metadata['onnx_sha256']
    assert metadata['max_abs_error']<.001
    assert session().get_providers()==['CPUExecutionProvider']

def test_pencil_is_lines_on_paper_without_gray_photo_base():
    im=np.full((180,240,3),220,np.uint8)
    cv2.rectangle(im,(70,40),(170,140),(45,45,45),3)
    p=Params(outline=1,pencil_shade=0,paper=0,protect=False)
    result=process(im,STYLES[3],p,cache=None)
    assert result[10:30,10:30].mean()>252
    assert np.percentile(result[35:145,65:175],2)<100
    assert result[70:110,95:145].mean()>252

def test_black_flat_field_does_not_become_false_ink():
    for value in (0,90,255):
        im=np.full((80,100,3),value,np.uint8)
        result=process(im,STYLES[3],Params(pencil_shade=0,paper=0,protect=False),cache=None)
        assert result.min()>=254

def test_flow_strokes_are_repeatable_and_not_global_screen_lines():
    im=np.zeros((96,128,3),np.uint8);cv2.circle(im,(64,48),30,(200,170,140),-1)
    _,_,tone,theta=flow_lines(im)
    a=graphite_strokes(tone,theta,np.ones_like(tone));b=graphite_strokes(tone,theta,np.ones_like(tone))
    assert np.array_equal(a,b) and np.isfinite(a).all()
    assert np.std(a)>.01
    assert not np.allclose(a,np.roll(a,4,axis=1))

def test_anime_export_does_not_mix_original_texture_back_in():
    yy,xx=np.mgrid[:1040,:1120]
    im=np.stack([130+35*np.sin(xx/70),140+30*np.cos(yy/80),150+20*np.sin((xx+yy)/100)],axis=2).astype(np.uint8)
    p=Params(protect=False)
    a=process(im,STYLES[0],p,quality='标准')
    b=process(im,STYLES[0],p,quality='精细')
    # Only interpolation changes; fine is not photographic detail restoration.
    assert np.abs(a.astype(float)-b.astype(float)).mean()<2

def test_face_refinement_rejects_tiny_faces_and_caps_cpu_work(monkeypatch):
    from cartoonify import anime
    calls=[]
    def fake_infer(image):
        calls.append(image.shape)
        return np.full_like(image,200 if image.shape[:2]==(512,512) else 30)
    monkeypatch.setattr(anime,'_infer',fake_infer)
    image=np.zeros((600,800,3),np.uint8)
    boxes=[(0,0,400,100),(10,10,30,30),(50,50,90,100),(250,50,90,100),(450,50,90,100),(50,400,90,100)]
    info={};result=anime.stylize(image,boxes,info)
    assert len(calls)==4 and info['refined_faces']==3
    assert np.all(result[0,-1]==30) and result[95,95].mean()>190
    assert np.all(result[445,95]==30)

def test_real_half_body_face_refinement_shares_preview_export_structure():
    from cartoonify.core import load_image,resize
    root=Path(__file__).resolve().parents[1]
    image=load_image(root/'samples/23_portrait.jpg');info={}
    preview=process(image,STYLES[0],preview=True,diagnostics=info)
    full=process(image,STYLES[0])
    assert info['anime_refined_faces']>=1
    assert preview.shape[:2]==resize(image,1000).shape[:2]
    assert np.abs(preview.astype(float)-resize(full,1000).astype(float)).mean()<4
