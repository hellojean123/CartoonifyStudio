import numpy as np
from cartoonify.progressive_pencil import regional_plan
def plan(dark,contour=None):
    a=np.full((70,90),dark,np.float32);z=np.zeros_like(a)
    return regional_plan(1-a,z,a,z,z,np.ones_like(a),contours=contour)
def test_round_feedback_and_decreasing_brush_width():
    p=plan(.6)
    assert len(p['round_metrics'])==4
    for r in p['round_metrics']:
        assert r['mean_positive_error_after']<=r['mean_positive_error_before']+1e-6
    assert np.median(p['width'][p['round']==0])>np.median(p['width'][p['round']==3])*2
def test_finished_regions_do_not_receive_redundant_strokes():
    assert len(plan(.6,np.full((70,90),3,np.float32))['width'])==0
    assert len(plan(0)['width'])==0
def test_same_input_repeatable_and_small_images_work():
    assert plan(.5)['signature']==plan(.5)['signature']
    a=np.ones((1,1),np.float32);p=regional_plan(a,a,a,a,a,a)
    assert np.isfinite(p['control']).all()
