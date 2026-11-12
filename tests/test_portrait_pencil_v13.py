import numpy as np
from cartoonify.portrait_pencil import portrait_detail
def prepared(flat=False):
    a=np.full((120,180,3),180,np.uint8)
    if not flat:
        a[47:54,47:68]=35;a[47:54,112:133]=35
        a[48:51,55:58]=250;a[48:51,120:123]=250
    return dict(guide=a,mask=np.ones((120,180),np.float32),boxes=[(20,10,140,100)],
                landmarks=[(np.array([20,10,140,100]),np.array([[58,50],[123,50],[90,70],[70,90],[110,90]],np.float32))])
def test_supported_eyes_preserve_highlight_and_dark_shape():
    d=portrait_detail(prepared())
    assert d['eyes_refined']==2
    assert d['eye_mask'][50,58]>.8 and d['eye_mask'][100,90]==0
    assert d['eye_drawing'][50,64]>d['eye_drawing'][49,56]+.3
def test_uniform_eye_roi_is_not_invented():
    d=portrait_detail(prepared(True))
    assert d['eyes_refined']==0 and not d['eye_mask'].any()
def test_missing_landmarks_are_safe_and_finite():
    p=prepared();p['landmarks']=[];d=portrait_detail(p)
    assert not d['eye_mask'].any()
    assert all(np.isfinite(d[k]).all() for k in ('eye_mask','eye_drawing','hair_boundary'))

def test_dark_eye_does_not_gain_an_artificial_white_patch():
    p=prepared();p['guide']=(p['guide'].astype(float)*.35).astype(np.uint8)
    d=portrait_detail(p)
    assert d['eyes_refined']==0 and not d['eye_mask'].any()
