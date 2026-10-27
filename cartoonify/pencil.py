"""CPU pencil: selective contours, tonal mass and finite graphite strokes.

Original implementation inspired by Lu/Xu/Jia (NPAR 2012) layer separation
and flow-aligned DoG literature; not a reproduction of either paper.
No downloaded texture, learned weights, or random photo-texture convolution.
"""
import cv2
import numpy as np


def _blur(a, sigma):
    return cv2.GaussianBlur(a, (0, 0), max(.35, float(sigma)))


def _contours(gray, smooth):
    """Keep persistent shape edges; reject small, weak texture responses."""
    sigma=.85+smooth*.055
    fine=_blur(gray,sigma)
    broad=_blur(gray,sigma*2.5)
    gx=cv2.Sobel(fine,cv2.CV_32F,1,0)/8
    gy=cv2.Sobel(fine,cv2.CV_32F,0,1)/8
    jxx=_blur(gx*gx,3.5);jyy=_blur(gy*gy,3.5);jxy=_blur(gx*gy,3.5)
    coherence=np.sqrt((jxx-jyy)**2+4*jxy*jxy)/(jxx+jyy+1e-6)
    theta=.5*np.arctan2(2*jxy,jxx-jyy)+np.pi/2
    yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]].astype(np.float32)
    response=np.maximum(broad-fine-.004,0)
    # Follow the direction at each integration step, rather than extending the
    # starting tangent as a straight ray through changing image structure.
    total=response.copy();weight=1.
    for sign in (-1,1):
        px=xx.copy();py=yy.copy();tx=np.cos(theta)*sign;ty=np.sin(theta)*sign
        for step in range(1,5):
            px+=tx;py+=ty
            nx=cv2.remap(np.cos(theta),px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            ny=cv2.remap(np.sin(theta),px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            flip=np.where(nx*tx+ny*ty<0,-1.,1.)
            norm=np.sqrt(nx*nx+ny*ny)+1e-6;tx=nx*flip/norm;ty=ny*flip/norm
            a=np.exp(-step*step/8)
            total+=cv2.remap(response,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)*a;weight+=a
    inner=np.clip(total/weight*22,0,1)*(.25+.75*coherence)
    # Larger-scale edges suppress isolated pores/leaves without requiring every
    # object to have a closed black outline.
    shape=_blur(gray,1.8+smooth*.12)
    edges=cv2.Canny(np.round(shape*255).astype(np.uint8),35,90,L2gradient=True)
    n,labels,stats,_=cv2.connectedComponentsWithStats(edges,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=max(5,round(max(gray.shape)*.012));keep[0]=False
    outer=np.clip(_blur(keep[labels].astype(np.float32),.55)*2.0,0,1)
    return outer,inner,theta,coherence


def graphite_strokes(tone, theta, coherence):
    """Short, tapered, jittered strokes; dark regions get a second hatch layer.

    The seed and canonical coordinates are fixed. Tone controls density, not
    local image variance: smooth cheeks and dark clothing retain their shade.
    """
    h,w=tone.shape;scale=max(h,w)/1000
    spacing=max(2.4,3.8*scale)
    rng=np.random.default_rng(843)
    yy,xx=np.mgrid[0:h:spacing,0:w:spacing]
    xx=np.clip(xx.ravel()+rng.uniform(-spacing*.45,spacing*.45,xx.size),0,w-1)
    yy=np.clip(yy.ravel()+rng.uniform(-spacing*.45,spacing*.45,yy.size),0,h-1)
    ix=xx.astype(int);iy=yy.astype(int)
    # Smooth the doubled-angle field to respect its pi-periodicity.
    vx=_blur(np.cos(2*theta)*coherence,14)
    vy=_blur(np.sin(2*theta)*coherence,14)
    field=.5*np.arctan2(vy,vx)
    reliability=_blur(coherence,8)
    angle=np.where(reliability[iy,ix]>.62,field[iy,ix],-.65)
    angle+=rng.uniform(-.16,.16,len(xx))
    density=np.clip((.96-tone[iy,ix])/.90,0,1)
    strokes=np.zeros((h,w),np.uint8)
    for cross in (False,True):
        chance=np.clip((density-.50)*1.2,0,.55) if cross else .13+density*.66
        selected=np.flatnonzero(rng.random(len(xx))<chance)
        lengths=rng.uniform(max(3.,7*scale),max(5.,16*scale),len(selected))
        strengths=rng.integers(85,205,len(selected))
        a=angle[selected]+(1.10 if cross else 0.)
        delta=np.stack((np.cos(a)*lengths,np.sin(a)*lengths),axis=-1)
        centers=np.stack((xx[selected],yy[selected]),axis=-1)
        points=np.round(centers[:,None,:]+delta[:,None,:]*np.array([-.5,-.22,.22,.5])[None,:,None]).astype(np.int32)
        # Quantized pressure groups let OpenCV draw many paths per call instead
        # of running per-stroke trigonometry and Python drawing loops.
        bins=strengths//16
        for level in np.unique(bins):
            group=points[bins==level]
            for j,pressure in enumerate((.52,1.,.65)):
                cv2.polylines(strokes,list(group[:,j:j+2,:]),False,int((level*16+8)*pressure),1,cv2.LINE_AA)
    return _blur(strokes.astype(np.float32)/255,.62)


def pencil_layers(prepared,p):
    rgb=prepared['blur'];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    outer,inner,theta,coherence=_contours(gray,p.smooth)
    face=prepared['mask'];features=prepared['features']
    # Selective simplification is continuous: no hard face-box boundaries.
    focus=np.ones_like(gray)
    if p.protect and np.any(face):
        skin=prepared['skin']*(1-features*.8)
        inner*=1-skin*p.face_strength*.85
        outer*=1-skin*p.face_strength*.35
        # An enlarged soft focus zone reduces background detail, not its tones.
        focus=_blur(face,max(2.,max(gray.shape)*.08))
        focus=np.clip(focus*2.,0,1)
        inner*=.48+.52*focus
        outer*=.72+.28*focus
    tone=_blur(gray,2.5+p.smooth*.5)
    density=np.clip((.86-tone)/.86,0,1)**1.65
    if p.protect:
        # A fixed luminance curve erased form on pale/brightly lit faces.
        # Estimate a separate highlight reference per face, then retain its
        # broad shadow differences without sharpening photographic pores.
        for x,y,bw,bh in prepared['boxes']:
            x0=max(0,x);y0=max(0,y);x1=min(gray.shape[1],x+bw);y1=min(gray.shape[0],y+bh)
            if x1<=x0 or y1<=y0:continue
            crop=tone[y0:y1,x0:x1];skin=prepared['skin'][y0:y1,x0:x1]
            selected=crop[skin>.25]
            if selected.size<16:continue
            reference=float(np.quantile(selected,.85))
            form=np.clip((reference-tone)/(reference+.05),0,1)**.8*.80
            # The existing soft face mask avoids a new rectangular boundary.
            zone=np.zeros_like(gray);zone[y0:y1,x0:x1]=1
            zone=_blur(zone,max(1.,bw*.06))*face*p.face_strength
            density+=np.maximum(form-density,0)*zone
    density*=1-p.face_strength*(1-focus)*.72 if p.protect else 1.
    strokes=graphite_strokes(tone,theta,coherence)
    # Optical absorption describes pigments, rather than tinting the paper.
    color=_blur(rgb.astype(np.float32)/255,3)
    absorption=1-color
    absorption/=(np.max(absorption,axis=2,keepdims=True)+.05)
    absorption=.22+.78*absorption
    return {'base':np.full_like(rgb,255),'coarse':outer,'internal':inner,
            'tone':tone,'density':density,'strokes':strokes,
            'pigment':absorption.astype(np.float32),'anime_faces':0}


def paper_grain(xx,yy):
    """Nonperiodic coordinate hash; stable across tiles/output resolutions."""
    x=np.floor(xx).astype(np.uint32);y=np.floor(yy).astype(np.uint32)
    value=(x*np.uint32(374761393)+y*np.uint32(668265263))
    value=(value^(value>>np.uint32(13)))*np.uint32(1274126177)
    value^=value>>np.uint32(16)
    return (value&np.uint32(65535)).astype(np.float32)/65535-.5
