"""Brush behavior tests: geometric pressure, buildup, native rasterization."""
import cv2
import numpy as np
from cartoonify.strokes import make_plan,sample_brush,rasterize


def line_plan(width=.45,profile=0):
    c=np.array([[[12,30],[35,25],[70,36],[92,30]]],np.float32)
    return {'control':c,'width':np.array([width],np.float32),'pressure':np.array([.85],np.float32),
            'dose':np.array([1.2],np.float32),'profile':np.array([profile],np.float32),
            'phase':np.array([.8],np.float32),'length':np.array([80],np.float32),
            'shape':np.array([64,104],np.int32),'bounds':np.concatenate((c.min(1),c.max(1)),axis=1)}


def test_pressure_and_width_are_continuous_and_finite_at_endpoints():
    with np.errstate(all='raise'):
        _,width,dose=sample_brush(line_plan(),np.array([0]),np.linspace(0,1,301,dtype=np.float32))
    assert np.isfinite(width).all() and np.isfinite(dose).all()
    assert width.max()>width.min()*1.3
    assert np.max(np.abs(np.diff(dose[0])))<.06
    profiles=[sample_brush(line_plan(profile=i),np.array([0]),np.linspace(0,1,31))[2] for i in range(5)]
    assert all(not np.allclose(profiles[i],profiles[j]) for i in range(5) for j in range(i))


def test_pencil_deposition_accumulates_without_erasing_and_saturates():
    plan=line_plan();single=rasterize(plan,(64,104))
    repeated={k:np.concatenate([v]*3) if k!='shape' else v for k,v in plan.items()}
    triple=rasterize(repeated,(64,104))
    assert np.all(triple>=single-1e-6)
    assert triple.max()>single.max()+.10 and triple.max()<2.5
    reverse={k:v[::-1].copy() if k!='shape' else v for k,v in repeated.items()}
    assert np.allclose(triple,rasterize(reverse,(64,104)),atol=2e-6)


def test_subpixel_width_and_darkness_are_independent_controls():
    thin=line_plan(.18);wide=line_plan(1.2)
    a=rasterize(thin,(64,104));b=rasterize(wide,(64,104))
    assert b.sum()>a.sum()*2
    thin['dose']*=3
    # Compare at a resolution where the fine nib is resolved, rather than
    # confusing subpixel coverage with intrinsic graphite darkness.
    dark_thin=rasterize(thin,(256,416))
    assert dark_thin.max()>rasterize(wide,(256,416)).max()
    assert not np.array_equal(a,b)


def test_native_resolution_rasterization_and_global_tile_coordinates():
    plan=line_plan(.35)
    low=rasterize(plan,(64,104));full=rasterize(plan,(256,416))
    enlarged=cv2.resize(low,(416,256),interpolation=cv2.INTER_CUBIC)
    assert np.abs(full-enlarged).mean()>.001
    assert abs(full.sum()/16-low.sum())/low.sum()<.15
    tiles=np.empty_like(full)
    for y in range(0,256,71):
        for x in range(0,416,89):
            bounds=(y,min(256,y+71),x,min(416,x+89))
            tiles[bounds[0]:bounds[1],bounds[2]:bounds[3]]=rasterize(plan,(256,416),bounds)
    assert np.allclose(full,tiles,atol=1e-6)


def test_generated_plan_has_varied_widths_and_repeatable_paths():
    tone=np.full((100,160),.3,np.float32);angle=np.zeros_like(tone)
    a=make_plan(tone,angle,np.zeros_like(tone));b=make_plan(tone,angle,np.zeros_like(tone))
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert np.percentile(a['width'],90)>np.percentile(a['width'],10)*1.8
    assert len(np.unique(a['profile']))==5


def test_invalid_face_coordinates_are_rejected_before_conversion(monkeypatch):
    from cartoonify.core import face_mask
    valid=np.array([20,20,40,50,30,35,50,35,40,45,30,58,50,58,.95],np.float32)
    nan=valid.copy();nan[0]=np.nan
    huge=valid.copy();huge[2]=1e30
    negative=valid.copy();negative[3]=-5
    bad_landmark=valid.copy();bad_landmark[5]=np.inf
    huge_landmark=valid.copy();huge_landmark[4]=1e30
    class Detector:
        def detect(self,image):return None,np.stack([nan,huge,negative,bad_landmark,huge_landmark,valid])
    class Factory:
        @staticmethod
        def create(*args,**kwargs):return Detector()
    monkeypatch.setattr(cv2,'FaceDetectorYN',Factory)
    with np.errstate(all='raise'):
        mask,boxes,points=face_mask(np.full((100,100,3),120,np.uint8),details=True)
    assert len(points)==1 and boxes==[(20,20,40,50)] and np.isfinite(mask).all()
