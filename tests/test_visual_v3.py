"""Structural checks, deliberately not claiming to measure aesthetics."""
from dataclasses import replace
import cv2
import numpy as np
from cartoonify.core import Params,STYLES,process,portrait_regions,merge_small_regions

def test_pencil_preserves_shadow_without_paper():
    im=np.full((160,240,3),225,np.uint8);im[:,120:]=45
    p=Params(outline=0,paper=0,pencil_shade=.5,protect=False)
    out=process(im,STYLES[3],p,cache=None)
    assert out[:,20:80].mean()-out[:,160:220].mean()>70

def test_features_not_suppressed_as_cheeks():
    im=np.full((200,200,3),(170,120,90),np.uint8)
    skin,features=portrait_regions(im,np.ones((200,200),np.float32),[(20,10,160,180)])
    assert skin[115,65]>.8
    assert features[82,68]>features[115,65]+.4

def test_region_merge_keeps_large_shapes():
    labels=np.zeros((60,80),np.uint8);labels[:,40:]=1;labels[20:23,20:23]=1
    result=merge_small_regions(labels,2,16)
    assert np.all(result[20:23,20:23]==0)
    assert np.all(result[:,40:]==1)

def test_flat_export_does_not_restore_photo_texture():
    yy,xx=np.mgrid[:1100,:1200];v=(160+12*np.sin(xx*1.8)*np.sin(yy*1.7)).astype(np.uint8)
    im=np.repeat(v[...,None],3,axis=2);p=Params(protect=False,outline=0)
    standard=process(im,STYLES[1],p,quality='标准')
    fine=process(im,STYLES[1],p,quality='精细')
    assert np.array_equal(standard,fine)
