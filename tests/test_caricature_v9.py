from dataclasses import replace,fields
import cv2,numpy as np,pytest
from cartoonify.caricature import deformation_map,FEATURES
from cartoonify.core import process,Params,STYLES,normalized
from cartoonify.cache import ByteLRU


def face():
    return ((70,35,160,220),np.float32([[115,110],[185,110],[150,155],[125,205],[175,205]]))


@pytest.mark.parametrize('feature',FEATURES)
def test_maps_are_monotonic_and_zero_is_identity(feature):
    shape=(300,300,3);yy,xx=np.mgrid[:300,:300].astype(np.float32)
    mx,my,info=deformation_map(shape,[face()],feature,0)
    assert np.array_equal(mx,xx) and np.array_equal(my,yy) and info['warped_faces']==0
    mx,my,info=deformation_map(shape,[face()],feature,1)
    determinant=np.gradient(mx,axis=1)*np.gradient(my,axis=0)-np.gradient(mx,axis=0)*np.gradient(my,axis=1)
    assert determinant.min()>.04 and np.isfinite(mx).all() and np.isfinite(my).all()
    assert np.max(np.abs(mx-xx))+np.max(np.abs(my-yy))>10
    assert np.max(np.abs(mx[:10,:10]-xx[:10,:10]))<.0001


def test_nose_is_really_enlarged_and_warp_cap_deduplicates():
    im=np.full((300,300),255,np.uint8);cv2.circle(im,(150,155),10,0,-1)
    mx,my,info=deformation_map(im.shape,[face(),face()],'鼻子',.9)
    result=cv2.remap(im,mx,my,cv2.INTER_LINEAR)
    assert np.count_nonzero(result<100)>np.count_nonzero(im<100)*2
    assert info['warped_faces']==1
    tiny=((0,0,20,20),None)
    assert deformation_map(im.shape,[tiny],'鼻子',1)[2]['warped_faces']==0


def test_modes_and_strength_invalidate_only_comic_layers():
    im=np.full((160,200,3),170,np.uint8);cv2.circle(im,(100,80),20,(20,30,40),-1)
    p=Params(face_boxes='[[0.2,0.05,0.6,0.9]]',caricature_strength=0,dot_opacity=0)
    cache=ByteLRU(30*1024**2);a=process(im,STYLES[2],p,cache=cache)
    before=cache.stats()['misses'];b=process(im,STYLES[2],replace(p,caricature_strength=1),cache=cache)
    assert cache.stats()['misses']==before+1 and np.abs(a.astype(float)-b.astype(float)).mean()>.5
    before=cache.stats()['misses'];process(im,STYLES[2],replace(p,caricature_strength=1,outline=.1),cache=cache)
    assert cache.stats()['misses']==before
    with pytest.raises(ValueError):normalized(replace(p,caricature_strength=float('nan')),STYLES[2])
    with pytest.raises(ValueError):normalized(replace(p,caricature_feature='unknown'),STYLES[2])


def test_cancel_and_invalid_detection_do_not_create_warps():
    with pytest.raises(InterruptedError):deformation_map((100,100,3),[face()],'鼻子',1,lambda:True)
    mx,my,info=deformation_map((100,100,3),[((0,0,float('nan'),100),None)],'鼻子',1)
    assert info['warped_faces']==0 and np.isfinite(mx).all()


def test_ui_defaults_disable_halftone_and_reset_controls():
    from app import build_app,defaults
    values=defaults(STYLES[2]);count=len(fields(Params))
    assert len(values)==count+2 and values[11]==0
    assert values[count-2:count]==('鼻子',.75)
    app=build_app();assert any(getattr(b,'label',None)=='重点夸张哪里' for b in app.blocks.values())


def test_portrait_protection_maps_follow_the_deformed_face():
    from cartoonify.caricature import caricature_layers
    im=np.full((300,300,3),180,np.uint8);mask=np.zeros((300,300),np.float32)
    cv2.circle(mask,(150,155),10,1,-1)
    prepared={'guide':im,'blur':im,'mask':mask,'skin':mask,'features':mask,'landmarks':[face()],'boxes':[face()[0]]}
    layers=caricature_layers(prepared,Params(caricature_strength=.9))
    assert layers['mask_override'].sum()>mask.sum()*2
    assert np.array_equal(layers['mask_override'],layers['features_override'])
