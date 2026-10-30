"""CPU adaptation of Tong et al., AAAI 2021: ETF, gray layers, ordered strokes.

Independently implemented from the published method and official CC0 source.
See pencil_v11_report.md for algorithm differences and source provenance.
"""
import hashlib
import cv2
import numpy as np


def checkpoint(cancel):
    if cancel and cancel():raise InterruptedError('处理已取消')


def tangent_flow(gray,cancel=None,iterations=15):
    h,w=gray.shape;scale=min(1.,512/max(h,w));size=(max(1,round(w*scale)),max(1,round(h*scale)))
    small=cv2.resize(gray,size,interpolation=cv2.INTER_AREA)
    gx=cv2.Sobel(small,cv2.CV_32F,1,0,ksize=5)/128
    gy=cv2.Sobel(small,cv2.CV_32F,0,1,ksize=5)/128
    magnitude=cv2.magnitude(gx,gy);g=magnitude/max(float(magnitude.max()),1e-6)
    norm=np.maximum(magnitude,1e-6);vx=-gy/norm;vy=gx/norm
    weak=magnitude<1e-5;vx[weak]=np.float32(2**-.5);vy[weak]=-np.float32(2**-.5)
    radius=3;hh,ww=small.shape;gp=np.pad(g,radius,mode='edge')
    for _ in range(iterations):
        checkpoint(cancel);xp=np.pad(vx,radius,mode='edge');yp=np.pad(vy,radius,mode='edge')
        ax=np.zeros_like(vx);ay=np.zeros_like(vy)
        for dy in range(-radius,radius+1):
            for dx in range(-radius,radius+1):
                nx=xp[radius+dy:radius+dy+hh,radius+dx:radius+dx+ww]
                ny=yp[radius+dy:radius+dy+hh,radius+dx:radius+dx+ww]
                ng=gp[radius+dy:radius+dy+hh,radius+dx:radius+dx+ww]
                dot=vx*nx+vy*ny
                weight=.5*(1+np.tanh(ng-g))*dot*ng
                ax+=nx*weight;ay+=ny*weight
        length=cv2.magnitude(ax,ay);valid=length>1e-8
        vx=np.where(valid,ax/np.maximum(length,1e-8),vx)
        vy=np.where(valid,ay/np.maximum(length,1e-8),vy)
    # Interpolate unoriented lines in double-angle coordinates, avoiding sign seams.
    c=cv2.resize(vx*vx-vy*vy,(w,h),interpolation=cv2.INTER_LINEAR)
    s=cv2.resize(2*vx*vy,(w,h),interpolation=cv2.INTER_LINEAR)
    theta=.5*np.arctan2(s,c)
    reliability=cv2.resize(cv2.GaussianBlur(g,(0,0),2),(w,h),interpolation=cv2.INTER_LINEAR)
    return theta,reliability


def rotate_map(shape,degrees):
    h,w=shape;matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),degrees,1)
    c=abs(matrix[0,0]);s=abs(matrix[0,1]);nw=max(1,int(np.ceil(w*c+h*s)));nh=max(1,int(np.ceil(h*c+w*s)))
    matrix[0,2]+=(nw-w)/2;matrix[1,2]+=(nh-h)/2
    return matrix,(nw,nh)


def ordered_plan(tone,theta,reliability,density,skin=None,focus=None,features=None,cancel=None):
    """Cumulative 10-level gray masks, 10 ETF directions, jittered scanline runs."""
    h,w=tone.shape;scale=max(h,w)/1000;scale=max(.25,scale)
    skin=np.zeros_like(tone) if skin is None else skin
    focus=np.ones_like(tone) if focus is None else focus
    features=np.zeros_like(tone) if features is None else features
    rng=np.random.default_rng(202111);directions=10;levels=10
    theta=np.where(reliability>.012,theta,-np.pi/4)
    direction=(np.floor((theta+np.pi/2)/(np.pi/directions)+.5).astype(int)%directions)
    target=np.clip(density*(.40+.60*focus),0,1)
    controls=[];widths=[];pressures=[];doses=[];profiles=[];phases=[];lengths=[];importance=[];orders=[]
    gx=cv2.Sobel(tone,cv2.CV_32F,1,0)/8;gy=cv2.Sobel(tone,cv2.CV_32F,0,1)/8;gradient=cv2.magnitude(gx,gy)
    spacing=max(1.25,4.0*scale)
    for d in range(directions):
        checkpoint(cancel);angle=-90+d*180/directions;matrix,size=rotate_map((h,w),angle)
        inv=cv2.invertAffineTransform(matrix)
        direction_mask=(direction==d).astype(np.uint8)
        dm=cv2.warpAffine(direction_mask,matrix,size,flags=cv2.INTER_NEAREST,borderValue=0)
        ink=cv2.warpAffine(target,matrix,size,flags=cv2.INTER_LINEAR,borderValue=0)
        for level in range(levels):
            checkpoint(cancel);threshold=(level+.6)/levels
            mask=(ink>threshold)&(dm>0)
            offset=rng.uniform(0,spacing)
            for y in np.arange(offset,size[1],spacing):
                iy=min(size[1]-1,int(y));row=mask[iy]
                changes=np.diff(np.pad(row.astype(np.int8),(1,1)));starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
                for start,end in zip(starts,ends):
                    if end-start<max(2,2*scale):continue
                    # Finite strokes stop at gray/direction boundaries; avoid kilometer-long strokes.
                    pos=float(start)
                    while pos<end-1:
                        stop=min(float(end-1),pos+rng.uniform(16,55)*scale)
                        if stop-pos<max(1.5,2*scale):break
                        xx=np.array([pos,pos+(stop-pos)/3,pos+2*(stop-pos)/3,stop],np.float32)
                        yy=np.full(4,y,np.float32);yy[1:3]+=rng.normal(0,.23*scale,2)
                        curve=np.stack([xx,yy,np.ones(4)],axis=1)@inv.T
                        center=curve[[0,3]].mean(0);cx=int(np.clip(center[0],0,w-1));cy=int(np.clip(center[1],0,h-1))
                        if not (0<=center[0]<w and 0<=center[1]<h):pos=stop+scale;continue
                        k=skin[cy,cx];density_here=target[cy,cx]
                        width=np.clip(rng.lognormal(np.log(.44),.29),.22,1.05)*scale*(1-.25*k)
                        controls.append(curve);widths.append(width);pressures.append(rng.uniform(.55,.95))
                        doses.append(rng.uniform(.75,1.18)*(1+.6*density_here));profiles.append(int(rng.integers(0,5)));phases.append(rng.uniform(0,2*np.pi));lengths.append(stop-pos)
                        importance.append(float((1-tone[cy,cx])*(gradient[cy,cx]+.02)+features[cy,cx]*.12));orders.append(level)
                        pos=stop+rng.uniform(.4,1.4)*scale
    control=np.asarray(controls,np.float32).reshape(-1,4,2)
    plan={'control':control,'width':np.asarray(widths,np.float32),'pressure':np.asarray(pressures,np.float32),'dose':np.asarray(doses,np.float32),
          'profile':np.asarray(profiles,np.float32),'phase':np.asarray(phases,np.float32),'length':np.asarray(lengths,np.float32),'shape':np.array([h,w],np.int32)}
    order=np.argsort(-np.asarray(importance),kind='stable')
    for key in ('control','width','pressure','dose','profile','phase','length'):plan[key]=plan[key][order]
    plan['importance']=np.asarray(importance,np.float32)[order];plan['gray_level']=np.asarray(orders,np.uint8)[order]
    plan['bounds']=np.concatenate((plan['control'].min(1),plan['control'].max(1)),axis=1)
    signature=hashlib.blake2b(digest_size=16)
    for key in ('control','width','pressure','dose','profile','phase','length','shape'):signature.update(plan[key].tobytes())
    plan['signature']=signature.hexdigest();return plan


def directional_structure(gray,smooth):
    """Lu-style directional gradient classification also used by the AAAI reference."""
    clean=cv2.medianBlur(gray.astype(np.float32),3)
    gradient=np.zeros_like(clean);gradient[:,:-1]+=abs(clean[:,1:]-clean[:,:-1]);gradient[:-1]+=abs(clean[1:]-clean[:-1])
    gradient=np.maximum(gradient-.003,0)
    radius=max(1,min(7,round(min(gray.shape)/85)));side=radius*2+1
    kernels=[];best=np.full(gray.shape,-1,np.float32);labels=np.zeros(gray.shape,np.uint8)
    for i in range(18):
        kernel=np.zeros((side,side),np.float32);kernel[radius]=1
        m=cv2.getRotationMatrix2D((radius,radius),i*10,1);kernel=cv2.warpAffine(kernel,m,(side,side));kernel/=max(float(kernel.sum()),1e-6)
        response=cv2.filter2D(gradient,-1,kernel);take=response>best;best[take]=response[take];labels[take]=i;kernels.append(kernel)
    stroke=np.zeros_like(gray)
    for i,kernel in enumerate(kernels):stroke+=cv2.filter2D(gradient*(labels==i),-1,kernel)
    return np.clip(stroke*gradient**.4*22,0,1)
