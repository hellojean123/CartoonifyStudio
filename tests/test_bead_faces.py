from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from cartoonify.beads import BeadParams,make_pattern,feature_cells,portrait_maps
from cartoonify.core import load_image,resize

ROOT=Path(__file__).parents[1]

def test_bright_and_dark_feature_boundaries_are_both_preserved():
    lab=np.zeros((4,4,3),np.float32);lab[:,:2,0]=20;lab[:,2:,0]=85
    protected=feature_cells(lab,np.ones((4,4),np.float32))
    assert protected[:,1].all() and protected[:,2].all()
    assert not feature_cells(lab,np.zeros((4,4),np.float32)).any()

def test_portrait_feature_error_improves_at_fixed_grid_and_palette_budget():
    image=load_image(ROOT/'samples/03_portrait.jpg');p=BeadParams()
    protected=make_pattern(image,p);plain=make_pattern(image,replace(p,preserve_features=False))
    h,w=protected.labels.shape;_,features,_=portrait_maps(resize(image,1000),(w,h));region=features>.3
    assert region.any() and len(protected.palette)<=p.colors
    reference=cv2.cvtColor(cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    def error(pattern):
        lab=cv2.cvtColor(pattern.palette[pattern.labels].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        return np.linalg.norm(lab-reference,axis=2)[region].mean()
    assert error(protected)<error(plain)*.8

def test_no_face_preserves_nonportrait_processing(monkeypatch):
    from cartoonify import beads
    im=np.random.default_rng(19).integers(0,256,(70,90,3),np.uint8)
    monkeypatch.setattr(beads,'portrait_maps',lambda work,shape:(np.zeros(shape[::-1],np.float32),np.zeros(shape[::-1],np.float32),[]))
    p=BeadParams(48,8)
    a=make_pattern(im,p);b=make_pattern(im,replace(p,preserve_features=False))
    assert np.array_equal(a.palette[a.labels],b.palette[b.labels])

def test_architectural_haar_false_positives_do_not_drive_face_palette():
    image=resize(load_image(ROOT/'samples/17_architecture.jpg'),1000)
    face,features,boxes=portrait_maps(image,(72,72))
    assert not boxes and not face.any() and not features.any()
