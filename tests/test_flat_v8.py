from dataclasses import replace
import cv2,numpy as np,pytest
from cartoonify.flat import exact_labels,spatial_labels,clean_islands,illustration_layers
from cartoonify.core import Params,process,STYLES


def test_exact_assignment_matches_brute_force():
    lab=np.random.default_rng(5).integers(0,256,(17,19,3),dtype=np.uint8)
    centers=np.float32([[40,130,160],[170,120,140],[230,128,130]])
    expected=np.sum((lab[...,None,:].astype(float)-centers)**2,axis=-1).argmin(-1)
    assert np.array_equal(exact_labels(lab,centers),expected)


def test_spatial_regularization_keeps_hard_boundaries():
    lab=np.full((40,60,3),(90,130,135),np.uint8);lab[:,30:]=(220,130,135)
    centers=np.float32([[90,130,135],[220,130,135]])
    initial=exact_labels(lab,centers);initial[10,10]=1
    out=spatial_labels(lab,centers,initial,8,np.zeros((40,60),np.float32))
    assert out[10,10]==0 and np.all(out[:,:30]==0) and np.all(out[:,30:]==1)


def test_component_cleanup_protects_small_facial_features():
    labels=np.zeros((30,30),np.uint8);labels[10:12,10:12]=1;labels[20,20]=1
    features=np.zeros((30,30),np.float32);features[10:12,10:12]=1
    result=clean_islands(labels,np.float32([[150,130,140],[10,130,140]]),features,10)
    assert result[10,10]==1 and result[20,20]==0


def test_color_budget_and_controls_change_real_output():
    yy,xx=np.mgrid[:120,:160]
    im=np.stack([xx*1.5+10,yy*1.7+25,(xx+yy)*.8],axis=-1).clip(0,255).astype(np.uint8)
    prepared={'blur':im,'features':np.zeros(im.shape[:2],np.float32),'mask':np.zeros(im.shape[:2],np.float32)}
    low=illustration_layers(prepared,Params(colors=6,smooth=3,protect=False))
    high=illustration_layers(prepared,Params(colors=20,smooth=3,protect=False))
    assert len(np.unique(low['base'].reshape(-1,3),axis=0))<=6
    assert np.abs(low['base'].astype(float)-high['base'].astype(float)).mean()>2
    a=process(im,STYLES[1],Params(protect=False),cache=None)
    b=process(im,STYLES[1],Params(protect=False),cache=None)
    assert np.array_equal(a,b)


def test_cancel_before_expensive_region_work():
    image=np.zeros((20,20,3),np.uint8);mask=np.zeros((20,20),np.float32)
    with pytest.raises(InterruptedError):illustration_layers({'blur':image,'features':mask,'mask':mask},Params(),lambda:True)
