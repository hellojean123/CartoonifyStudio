"""Small CPU graphite brush: cubic paths, continuous pressure and optical buildup.

Original implementation. References (ideas only, no third-party code/assets):
MyPaint pressure/flow separation, perfect-freehand pressure/taper geometry,
and Sousa/Buchanan graphite deposition. No claim of physical reproduction.
All paths live in analysis-image pixel coordinates; rasterization is performed
again at the requested output size, with bounded dab batches and tile memory.
"""
import hashlib
import cv2
import numpy as np


def make_plan(tone, theta, coherence, density=None, skin=None):
    h,w=tone.shape;scale=max(h,w)/1000
    density=np.clip(1-tone,0,1) if density is None else density
    skin=np.zeros_like(tone) if skin is None else skin
    rng=np.random.default_rng(619)
    spacing=max(1.6,2.35*scale)
    yy,xx=np.mgrid[0:h:spacing,0:w:spacing]
    x=np.clip(xx.ravel()+rng.uniform(-.48,.48,xx.size)*spacing,0,w-1)
    y=np.clip(yy.ravel()+rng.uniform(-.48,.48,yy.size)*spacing,0,h-1)
    ix=x.astype(int);iy=y.astype(int)
    vx=cv2.GaussianBlur(np.cos(2*theta)*coherence,(0,0),14)
    vy=cv2.GaussianBlur(np.sin(2*theta)*coherence,(0,0),14)
    reliability=cv2.GaussianBlur(coherence,(0,0),8)
    field=.5*np.arctan2(vy[iy,ix],vx[iy,ix])
    turn=(field+.62+np.pi/2)%np.pi-np.pi/2
    # A family of related hatching directions avoids fur-like swirls caused
    # by following every local lighting gradient in skin and flat clothing.
    direction=-.62+np.where(reliability[iy,ix]>.65,turn*.32,0)
    direction+=rng.normal(0,.055,len(x))
    d=density[iy,ix];k=skin[iy,ix]
    chunks=[]
    for cross in (False,True):
        chance=np.clip((d-.30)*.85,0,.50) if cross else .10+np.sqrt(d)*.75
        selected=np.flatnonzero((rng.random(len(x))<chance)&(d>.008))
        n=len(selected)
        angle=direction[selected]+(1.12 if cross else 0.)
        length=rng.uniform(max(2.,5.*scale),max(4.,15.*scale),n)
        length*=1-.22*k[selected]
        tangent=np.stack((np.cos(angle),np.sin(angle)),axis=-1)
        normal=np.stack((-np.sin(angle),np.cos(angle)),axis=-1)
        centers=np.stack((x[selected],y[selected]),axis=-1)
        bend=rng.normal(0,.025,n)*length
        control=np.stack([centers+tangent*(length*t)[:,None]+normal*(bend*b)[:,None]
                          for t,b in ((-.5,0),(-.18,1),(.18,1),(.5,0))],axis=1)
        # Mostly fine point work; broad, light side strokes are uncommon and
        # restricted to darker regions. Width and pigment dose are independent.
        width=np.clip(rng.lognormal(np.log(.40),.34,n),.16,.90)*max(.45,scale)
        side=(rng.random(n)<.07)&(d[selected]>.35)&(k[selected]<.3)
        width[side]*=1.8
        width*=1-.30*k[selected]
        pressure=rng.uniform(.35,.88,n)
        dose=rng.uniform(.65,1.25,n)*(1-.55*side)
        dose*=.65+np.sqrt(d[selected])*1.35
        dose*=np.clip(rng.lognormal(0,.30,n),.55,1.8)
        profile=rng.integers(0,5,n)
        phase=rng.uniform(0,2*np.pi,n)
        chunks.append((control,width,pressure,dose,profile,phase,length))
    arrays=[np.concatenate([c[i] for c in chunks],axis=0).astype(np.float32) for i in range(7)]
    plan=dict(zip(('control','width','pressure','dose','profile','phase','length'),arrays),shape=np.array([h,w],np.int32))
    plan['bounds']=np.concatenate((plan['control'].min(1),plan['control'].max(1)),axis=1)
    signature=hashlib.blake2b(digest_size=16)
    for name in ('control','width','pressure','dose','profile','phase','length','shape'):signature.update(plan[name].tobytes())
    plan['signature']=signature.hexdigest()
    return plan


def sample_brush(plan, indices, t):
    """Return path, varying FWHM width and pigment dose at normalized time t."""
    t=np.asarray(t,np.float32)[None,:];u=1-t
    c=plan['control'][indices]
    xy=c[:,0,None,:]*u[...,None]**3+3*c[:,1,None,:]*(u*u*t)[...,None]+3*c[:,2,None,:]*(u*t*t)[...,None]+c[:,3,None,:]*t[...,None]**3
    profile=plan['profile'][indices,None]
    envelope=np.where(profile==0,.18+.82*np.maximum(np.sin(np.pi*t),0)**.65,
             np.where(profile==1,.95-.75*t,
             np.where(profile==2,.20+.75*t,
             np.where(profile==3,.78+0*t,.40+.55*np.sin(np.pi*t*.85+.3)))))
    # Correlated, low-amplitude handling variation, not independent point noise.
    handling=1+.09*np.sin(t*2*np.pi+plan['phase'][indices,None])+.035*np.sin(t*5*np.pi+plan['phase'][indices,None]*1.7)
    pressure=np.clip(plan['pressure'][indices,None]*envelope*handling,.03,1)
    width=plan['width'][indices,None]*(.45+.85*np.sqrt(pressure))
    # Independent dose permits narrow/dark and broad/light marks.
    dose=plan['dose'][indices,None]*pressure
    return xy,width,dose


def _deposit(canvas,xy,width,dose,step,bounds):
    y0,y1,x0,x1=bounds
    # Pixel aperture and a small graphite footprint integrate subpixel coverage.
    # No final-image blur: fine marks retain their own coverage and taper.
    sigma=np.maximum(width/2.355,.025)
    variance=sigma*sigma+np.float32(1/8)
    radius=np.ceil(2.5*np.sqrt(variance)).astype(np.int32)
    amplitude=dose*step*sigma/(np.float32(2.5066283)*variance)
    for r in np.unique(radius):
        ids=np.flatnonzero(radius==r)
        offsets=np.stack(np.meshgrid(np.arange(-r,r+1,dtype=np.int32),np.arange(-r,r+1,dtype=np.int32)),axis=-1).reshape(-1,2)
        batch=max(1,min(2048,262144//len(offsets)))
        for start in range(0,len(ids),batch):
            selected=ids[start:start+batch]
            centers=xy[selected];pixels=np.floor(centers+.5).astype(np.int32)[:,None,:]+offsets[None,:,:]
            dx=pixels[...,0].astype(np.float32)-centers[:,None,0];dy=pixels[...,1].astype(np.float32)-centers[:,None,1]
            weight=np.exp(-(dx*dx+dy*dy)/(2*variance[selected,None]))*amplitude[selected,None]
            inside=(pixels[...,0]>=x0)&(pixels[...,0]<x1)&(pixels[...,1]>=y0)&(pixels[...,1]<y1)
            index=(pixels[...,1]-y0)*(x1-x0)+pixels[...,0]-x0
            np.add.at(canvas.ravel(),index[inside],weight[inside].astype(np.float32))


def rasterize(plan,target_shape,bounds=None,cancel=None):
    """Rasterize a global vector plan into a target tile as optical depth.

    Repeated marks add material. Output saturation is smooth and monotonic;
    a later light stroke never erases a previous dark stroke.
    """
    h,w=target_shape[:2]
    bounds=(0,h,0,w) if bounds is None else bounds
    y0,y1,x0,x1=bounds;canvas=np.zeros((y1-y0,x1-x0),np.float32)
    ah,aw=plan['shape'];ratio=np.array([w/aw,h/ah],np.float32)
    scale=float(np.sqrt(ratio.prod()))
    bounds_low=plan['bounds'][:,:2];bounds_high=plan['bounds'][:,2:]
    margin=plan['width']*scale*2+2
    low=(bounds_low+.5)*ratio-.5-margin[:,None]
    high=(bounds_high+.5)*ratio-.5+margin[:,None]
    selected=np.flatnonzero((low[:,0]<x1)&(high[:,0]>=x0)&(low[:,1]<y1)&(high[:,1]>=y0))
    # Sampling remains subpixel at the final resolution; a 4K export does not
    # enlarge the 1000-pixel graphite bitmap.
    counts=np.maximum(8,np.ceil(plan['length'][selected]*scale/.72/8).astype(np.int32)*8)
    for count in np.unique(counts):
        indices=selected[counts==count];t=np.linspace(0,1,int(count),dtype=np.float32)
        per_batch=max(1,8192//int(count))
        for start in range(0,len(indices),per_batch):
            if cancel is not None and cancel():raise InterruptedError('处理已取消')
            ids=indices[start:start+per_batch]
            xy,width,dose=sample_brush(plan,ids,t)
            xy=(xy+.5)*ratio-.5;width*=scale
            step=np.repeat(plan['length'][ids,None]*scale/(count-1),count,axis=1)
            step[:,[0,-1]]*=.5
            _deposit(canvas,xy.reshape(-1,2),width.ravel(),dose.ravel(),step.ravel(),bounds)
    return 2.5*(-np.expm1(-canvas/2.5))
