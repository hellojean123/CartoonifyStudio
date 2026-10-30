"""Behavior tests for ETF direction and grayscale-guided sequential drawing."""
import numpy as np
import pytest
from cartoonify.flow_pencil import tangent_flow,ordered_plan
from cartoonify.strokes import rasterize

def test_etf_follows_edges_and_is_finite_on_constant_images():
    gray=np.tile(np.linspace(0,1,100,dtype=np.float32),(80,1))
    theta,reliability=tangent_flow(gray,iterations=3)
    assert np.mean(np.abs(np.sin(theta[:,20:80])))>.95
    for shape in [(1,1),(1,16),(16,1),(40,60)]:
        t,r=tangent_flow(np.ones(shape,np.float32),iterations=2)
        assert t.shape==shape and np.isfinite(t).all() and np.isfinite(r).all()

def test_gray_layers_white_space_order_and_determinism():
    tone=np.tile(np.linspace(.1,1,140,dtype=np.float32),(100,1))
    theta=np.full_like(tone,-.6);reliability=np.ones_like(tone)
    density=np.clip((.86-tone)/.86,0,1)
    a=ordered_plan(tone,theta,reliability,density)
    b=ordered_plan(tone,theta,reliability,density)
    assert a['signature']==b['signature'] and len(a['width'])>0
    assert np.all(np.diff(a['importance'])<=0)
    assert len(np.unique(a['gray_level']))>3
    result=rasterize(a,tone.shape)
    assert result[:,:40].mean()>result[:,100:].mean()+.02
    assert result[:,-5:].max()<.001
    empty=ordered_plan(np.ones_like(tone),theta,reliability,np.zeros_like(tone))
    assert len(empty['width'])==0 and not rasterize(empty,tone.shape).any()

def test_flow_pipeline_cancellation_is_checked():
    with pytest.raises(InterruptedError):tangent_flow(np.zeros((30,40),np.float32),cancel=lambda:True)
    with pytest.raises(InterruptedError):ordered_plan(*[np.ones((30,40),np.float32)]*4,cancel=lambda:True)

def test_stroke_sequence_progress_only_adds_graphite():
    shape=(64,80);tone=np.full(shape,.25,np.float32)
    plan=ordered_plan(tone,np.zeros(shape,np.float32),np.ones(shape,np.float32),1-tone)
    cut=len(plan['width'])//2
    first={k:(v[:cut] if isinstance(v,np.ndarray) and k!='shape' else v) for k,v in plan.items()}
    assert np.all(rasterize(plan,shape)>=rasterize(first,shape)-1e-5)
