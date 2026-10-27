"""Structure-aware CPU line drawing and graphite strokes.

Separable structure-tensor flow filtering inspired by Kyprianidis & Doellner
(TPCG 2008); this is a NumPy/OpenCV implementation, not their source code.
"""
import cv2
import numpy as np

def flow_lines(rgb, smooth=6):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    f=cv2.GaussianBlur(gray,(0,0),.7)
    gx=cv2.Sobel(f,cv2.CV_32F,1,0)/8;gy=cv2.Sobel(f,cv2.CV_32F,0,1)/8
    jxx=cv2.GaussianBlur(gx*gx,(0,0),2.);jyy=cv2.GaussianBlur(gy*gy,(0,0),2.)
    jxy=cv2.GaussianBlur(gx*gy,(0,0),2.)
    theta=.5*np.arctan2(2*jxy,jxx-jyy)
    nx=np.cos(theta);ny=np.sin(theta)
    yy,xx=np.mgrid[:gray.shape[0],:gray.shape[1]].astype(np.float32)
    sigma=.85+smooth*.045
    narrow=np.zeros_like(gray);wide=np.zeros_like(gray);wn=ww=0.
    for offset in range(-4,5):
        sample=cv2.remap(f,xx+nx*offset,yy+ny*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        a=np.exp(-offset**2/(2*sigma**2));b=np.exp(-offset**2/(2*(sigma*1.65)**2))
        narrow+=sample*a;wide+=sample*b;wn+=a;ww+=b
    dog=narrow/wn-wide/ww
    coherent=np.zeros_like(gray);weight=0.
    for offset in range(-3,4):
        a=np.exp(-offset**2/8)
        coherent+=cv2.remap(dog,xx-ny*offset,yy+nx*offset,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)*a
        weight+=a
    fine=np.clip((-coherent/weight-.0035)*45,0,1)
    edges=cv2.Canny((f*255).astype(np.uint8),45,110,L2gradient=True)
    _,labels,stats,_=cv2.connectedComponentsWithStats(edges,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=max(3,round(max(gray.shape)*.006));keep[0]=False
    coarse=np.clip(cv2.GaussianBlur(keep[labels].astype(np.float32),(0,0),.48)*2.4,0,1)
    return coarse,fine,gray,theta

def coherent_strokes(tone,theta):
    """Deterministic line-integral noise follows local tangents, not a screen grid."""
    yy,xx=np.mgrid[:tone.shape[0],:tone.shape[1]].astype(np.float32)
    noise=np.random.default_rng(48).uniform(-1,1,tone.shape).astype(np.float32)
    noise=cv2.GaussianBlur(noise,(0,0),.4)
    tx=-np.sin(theta);ty=np.cos(theta);strokes=np.zeros_like(tone);total=0.
    for step in range(-7,8):
        weight=np.exp(-step*step/32)
        strokes+=cv2.remap(noise,xx+tx*step,yy+ty*step,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)*weight
        total+=weight
    local=cv2.GaussianBlur(tone,(0,0),9)
    variance=np.maximum(cv2.GaussianBlur(tone*tone,(0,0),9)-local*local,0)
    support=np.clip((np.sqrt(variance)-.008)*10,0,1)
    density=np.clip((.85-tone)/.85,0,1)**.85
    # Uniform photographic backgrounds become blank paper, not graphite noise.
    return np.clip((strokes/total+.07)*4,0,1)*density*support

def region_smooth(rgb,smooth=6,radius=None):
    """Soft-minimum four-sector Kuwahara; bounded analysis images only."""
    f=rgb.astype(np.float32)/255;radius=2+smooth//2 if radius is None else radius;k=radius+1
    candidates=[];variances=[]
    for dx,dy in ((0,0),(radius,0),(0,radius),(radius,radius)):
        m=cv2.boxFilter(f,-1,(k,k),anchor=(dx,dy),borderType=cv2.BORDER_REFLECT)
        v=np.maximum(cv2.boxFilter(f*f,-1,(k,k),anchor=(dx,dy),borderType=cv2.BORDER_REFLECT)-m*m,0).sum(2)
        candidates.append(m);variances.append(v)
    weights=1/(np.stack(variances)+.001)**2;weights/=weights.sum(0,keepdims=True)
    clean=sum(candidate*weight[...,None] for candidate,weight in zip(candidates,weights))
    return np.clip(clean*255,0,255).astype(np.uint8)
