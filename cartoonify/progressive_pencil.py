"""Contour-guided regional coarse-to-fine drawing with actual canvas feedback.

Original CPU implementation inspired by Hertzmann 1998 layered error feedback,
dualFace global/local structure, and Tong et al. 2021 edge tangent flow.
"""
import hashlib
import cv2
import numpy as np
from .strokes import rasterize


def seal(plan):
    c=plan['control'];plan['bounds']=np.concatenate((c.min(1),c.max(1)),axis=1)
    digest=hashlib.blake2b(digest_size=16)
    for k in ('control','width','pressure','dose','profile','phase','length','shape'):digest.update(plan[k].tobytes())
    plan['signature']=digest.hexdigest();return plan


def combine(parts,shape):
    keys=('control','width','pressure','dose','profile','phase','length','round')
    if not parts:
        p={k:np.empty((0,4,2) if k=='control' else (0,),np.float32) for k in keys}
    else:p={k:np.concatenate([part[k] for part in parts]) for k in keys}
    p['shape']=np.array(shape,np.int32);return seal(p)


def regional_plan(tone,theta,density,skin,features,focus,cancel=None,contours=None):
    h,w=tone.shape;scale=max(.3,max(h,w)/1000);rng=np.random.default_rng(121112)
    # Smooth the line field in skin so small gradients do not make fur-like marks.
    c=cv2.GaussianBlur(np.cos(2*theta),(0,0),12);s=cv2.GaussianBlur(np.sin(2*theta),(0,0),12)
    smooth=.5*np.arctan2(s,c);weight=skin*(1-features)*.85
    theta=.5*np.arctan2(np.sin(2*theta)*(1-weight)+np.sin(2*smooth)*weight,
                        np.cos(2*theta)*(1-weight)+np.cos(2*smooth)*weight)
    # Related hatching directions rather than copying every local wrinkle/texture.
    related=.32-.18*skin*(1-features)
    theta=(.5*np.arctan2(np.sin(-1.3)*(1-related)+np.sin(2*theta)*related,
                        np.cos(-1.3)*(1-related)+np.cos(2*theta)*related)).astype(np.float32)
    target=np.clip(density*(.18+.82*focus)*2.5,0,2.2).astype(np.float32)
    contour=np.zeros_like(tone) if contours is None else contours.astype(np.float32)
    current=contour.copy();parts=[];metrics=[]
    def sample(a,x,y):
        result=np.empty(len(x),np.float32)
        for start in range(0,len(x),16000):
            end=min(len(x),start+16000)
            result[start:end]=cv2.remap(a,x[start:end].astype(np.float32).reshape(1,-1),y[start:end].astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel()
        return result
    for stage,(spacing,length,width,fraction) in enumerate([(7.,32.,1.0,.35),(5.,22.,.65,.65),(3.8,14.,.40,.85),(2.8,9.,.25,1.)]):
        if cancel and cancel():raise InterruptedError('处理已取消')
        goal=target*fraction
        # Re-evaluate the already drawn canvas at this scale, not a fixed pre-sort.
        residual=np.maximum(goal-cv2.GaussianBlur(current,(0,0),max(.6,width*scale)),0)
        yy,xx=np.mgrid[0:h:max(1.2,spacing*scale),0:w:max(1.2,spacing*scale)]
        x=np.clip(xx.ravel()+rng.uniform(-.45,.45,xx.size)*spacing*scale,0,w-1)
        y=np.clip(yy.ravel()+rng.uniform(-.45,.45,yy.size)*spacing*scale,0,h-1)
        r=sample(residual,x,y);f=sample(features,x,y);k=sample(skin,x,y)
        keep=(r>.04)&(rng.random(len(x))<np.clip(r*2.8,.03,.95))
        if stage>=2:keep&=rng.random(len(x))<np.clip(.2+sample(focus,x,y)*.8+r,.2,1)
        x=x[keep];y=y[keep];r=r[keep];f=f[keep];k=k[keep];n=len(x)
        angle=sample(theta,x,y)+rng.normal(0,.045,n)
        if stage==3:angle+=.55*(1-np.exp(-np.maximum(sample(target,x,y)-.5,0)*2))*np.exp(-k*5)
        lengths=length*scale*rng.uniform(.65,1.25,n)*(1-.5*f)
        center=np.stack((x,y),axis=1);direction=np.stack((np.cos(angle),np.sin(angle)),axis=1)
        # Reject paths crossing strong tonal boundaries before depositing pigment.
        ends=center[:,None,:]+direction[:,None,:]*lengths[:,None,None]*np.array([-.5,.5],np.float32)[None,:,None]
        at_center=sample(tone,x,y)
        for side in (0,1):
            difference=np.abs(sample(tone,ends[:,side,0],ends[:,side,1])-at_center)
            shorten=.22+.78*np.exp(-(difference/.12)**2)
            ends[:,side]=center+(ends[:,side]-center)*shorten[:,None]
        tangent0=sample(theta,ends[:,0,0],ends[:,0,1]);tangent1=sample(theta,ends[:,1,0],ends[:,1,1])
        v0=np.stack((np.cos(tangent0),np.sin(tangent0)),axis=1);v1=np.stack((np.cos(tangent1),np.sin(tangent1)),axis=1)
        v0*=np.where((v0*direction).sum(1)<0,-1,1)[:,None];v1*=np.where((v1*direction).sum(1)<0,-1,1)[:,None]
        actual=np.linalg.norm(ends[:,1]-ends[:,0],axis=1)
        control=np.stack((ends[:,0],ends[:,0]+v0*actual[:,None]/3,ends[:,1]-v1*actual[:,None]/3,ends[:,1]),axis=1)
        part={'control':control.astype(np.float32),'width':(width*scale*rng.uniform(.7,1.3,n)*(1-.25*k)).astype(np.float32),
              'pressure':rng.uniform(.4,.9,n).astype(np.float32),'dose':np.clip(r*2.5,.12,1.5).astype(np.float32),
              'profile':rng.integers(0,5,n).astype(np.float32),'phase':rng.uniform(0,6.28,n).astype(np.float32),
              'length':actual.astype(np.float32),'round':np.full(n,stage,np.float32)}
        parts.append(part);plan=combine(parts,tone.shape);current=contour+rasterize(plan,tone.shape,cancel=cancel)
        metrics.append({'round':stage+1,'new_strokes':n,'mean_positive_error_before':float(residual.mean()),
                        'mean_positive_error_after':float(np.maximum(goal-cv2.GaussianBlur(current,(0,0),max(.6,width*scale)),0).mean())})
    plan['round_metrics']=metrics;return plan
