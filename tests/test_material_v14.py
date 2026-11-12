import numpy as np
from cartoonify.material_pencil import material_structure
from cartoonify.portrait_pencil import portrait_detail
def test_constant_material_has_no_invented_grooves():
    im=np.full((80,100,3),150,np.uint8)
    for skin in (np.ones((80,100),np.float32),np.zeros((80,100),np.float32)):
        relief,form=material_structure(im,skin)
        assert relief.max()<1e-6 and np.ptp(form)<1e-6
def test_skin_and_fabric_valleys_are_detected_without_blackening_whole_surface():
    yy,xx=np.mgrid[:100,:160]
    gray=(210-90*np.exp(-((xx-80)/5)**2)).astype(np.uint8)
    im=np.repeat(gray[...,None],3,axis=2)
    for skin in (np.ones(gray.shape,np.float32),np.zeros(gray.shape,np.float32)):
        relief,form=material_structure(im,skin)
        assert relief[:,75:85].max()>.02
        assert relief[:,:20].max()<.001 and form[:,80].mean()<form[:,10].mean()
def test_eye_influence_has_a_soft_spatial_falloff():
    import runpy
    from pathlib import Path
    prepared=runpy.run_path(str(Path(__file__).with_name("test_portrait_pencil_v13.py")))["prepared"]
    d=portrait_detail(prepared());a=d['eye_mask'][50]
    assert a[58]>.8 and 0<a[45]<a[50]<a[58]
