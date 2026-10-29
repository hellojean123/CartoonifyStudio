"""CPU geometric caricature and colored-ink rendering, no face synthesis."""
import json
import cv2
import numpy as np

FEATURES=('鼻子','眼睛','笑容','额头')


def feature_warps(box,points,feature,strength):
    x,y,bw,bh=map(float,box)
    if min(bw,bh)<50 or not .45<bw/bh<1.8:return []
    p=np.asarray(points,dtype=np.float32) if points is not None else np.float32([
        [x+bw*.32,y+bh*.4],[x+bw*.68,y+bh*.4],[x+bw*.5,y+bh*.61],
        [x+bw*.36,y+bh*.8],[x+bw*.64,y+bh*.8]])
    if p.shape!=(5,2) or not np.isfinite(p).all():return []
    if feature=='鼻子':return [(p[2,0],p[2,1],bw*.45,bh*.43,strength*1.2,1.)]
    if feature=='眼睛':return [(a,b,bw*.32,bh*.26,strength*.95,1.) for a,b in p[:2]]
    if feature=='笑容':
        a,b=p[3:5].mean(0);return [(a,b,bw*.55,bh*.31,strength*1.15,.45)]
    if feature=='额头':return [(float(p[:2,0].mean()),y+bh*.16,bw*.67,bh*.55,strength*1.2,1.)]
    raise ValueError('未知夸张特征')


def deformation_map(shape,faces,feature,strength,cancel=None):
    """Compact-support inverse maps; radial source distance increases monotonically."""
    if feature not in FEATURES:raise ValueError('未知夸张特征')
    if not np.isfinite(strength) or not 0<=strength<=1:raise ValueError('夸张程度须为 0～1')
    h,w=shape[:2];my,mx=np.mgrid[:h,:w].astype(np.float32);used=0;accepted=[]
    for box,points in sorted(faces,key=lambda f:float(f[0][2])*float(f[0][3]),reverse=True):
        if cancel and cancel():raise InterruptedError('处理已取消')
        values=np.asarray(box,dtype=np.float64)
        if values.shape!=(4,) or not np.isfinite(values).all() or np.max(np.abs(values))>4*max(h,w):continue
        x,y,bw,bh=values
        if x+bw<=0 or y+bh<=0 or x>=w or y>=h:continue
        if any(max(0,min(x+bw,a+c)-max(x,a))*max(0,min(y+bh,b+d)-max(y,b))/max(1,min(bw*bh,c*d))>.5 for a,b,c,d in accepted):continue
        warps=feature_warps(box,points,feature,strength)
        if not warps or strength==0:continue
        for cx,cy,rx,ry,amount,vertical in warps:
            dx=mx-cx;dy=my-cy;r2=(dx/rx)**2+(dy/ry)**2
            falloff=np.maximum(1-r2,0)**3
            mx=cx+dx/(1+amount*falloff)
            my=cy+dy/(1+amount*vertical*falloff)
        accepted.append(tuple(values));used+=1
        if used==3:break
    return mx,my,{'warped_faces':used,'feature':feature,'strength':float(strength),'max_faces':3}


def caricature_layers(prepared,p,cancel=None):
    from .core import face_mask,quantize
    from .npr import region_smooth,flow_lines
    guide=prepared['guide'];h,w=guide.shape[:2]
    landmarks=list(prepared.get('landmarks',()))
    if not p.protect and p.caricature_strength>0:
        _,_,landmarks=face_mask(guide,details=True)
    faces=[(box,points) for box,points in landmarks if min(box[2:])>=70]
    if p.face_boxes.strip():
        # Manual inputs may be present when smoothing protection is disabled.
        if not p.protect:face_mask(guide,p.face_boxes,details=True)
        for x,y,bw,bh in json.loads(p.face_boxes):faces.append(((round(x*w),round(y*h),round(bw*w),round(bh*h)),None))
    mx,my,info=deformation_map(guide.shape,faces,p.caricature_feature,p.caricature_strength,cancel)
    def warp(a):return cv2.remap(a,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    drawn=warp(prepared['blur'])
    regions=region_smooth(drawn,p.smooth,radius=3+p.smooth)
    features=warp(prepared['features'])
    regions=np.clip(regions*(1-features[...,None]*.75)+drawn*(features[...,None]*.75),0,255).astype(np.uint8)
    # Continuous colored-pencil/ink wash with a modest palette scaffold.
    palette=quantize(regions,min(48,max(12,p.colors*3)))
    base=np.clip(palette.astype(np.float32)*.60+regions.astype(np.float32)*.40,0,255).astype(np.uint8)
    hsv=cv2.cvtColor(base,cv2.COLOR_RGB2HSV);hsv[...,1]=np.clip(hsv[...,1].astype(float)*1.10,0,255).astype(np.uint8)
    base=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    line_guide=np.clip(regions*(1-features[...,None]*.8)+drawn*(features[...,None]*.8),0,255).astype(np.uint8)
    coarse,internal,tone,theta=flow_lines(line_guide,p.smooth)
    # Vary ink locally along image structures, rather than a uniform thick outline.
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    pressure=.72+.18*np.sin(xx*.047+yy*.031)+.10*np.cos(xx*.019-yy*.043)
    coarse*=pressure*.65;internal*=pressure*.45
    return {'base':base,'coarse':coarse,'internal':internal,'tone':tone,'strokes':np.zeros_like(tone),
            'mask_override':warp(prepared['mask']),'skin_override':warp(prepared['skin']),
            'features_override':warp(prepared['features']),'caricature':info}
