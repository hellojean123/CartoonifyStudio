"""CPU flat illustration: edge-aware regions and a spatially coherent palette."""
import cv2
import numpy as np


def spatial_labels(lab,centers,initial,smooth,features,cancel=None):
    """Red-black local optimization; O(HW) memory, never allocate H*W*K."""
    f=lab.astype(np.float32);labels=initial.copy();h,w=labels.shape
    parity=(np.arange(h,dtype=np.uint16)[:,None]+np.arange(w,dtype=np.uint16)[None,:])&1
    weights=[]
    directions=((1,0),(-1,0),(0,1),(0,-1))
    for dy,dx in directions:
        difference=f-np.roll(f,(dy,dx),(0,1))
        weight=np.exp(-np.sum(difference*difference,axis=2)/(2*16**2))
        if dy==1:weight[0]=0
        if dy==-1:weight[-1]=0
        if dx==1:weight[:,0]=0
        if dx==-1:weight[:,-1]=0
        weights.append(weight)
    strength=(12+smooth*2.4)**2*(1-features*.55)
    norm=np.sum(f*f,axis=2)
    for _ in range(2):
        for color_pass in (0,1):
            if cancel and cancel():raise InterruptedError('处理已取消')
            neighbors=[np.roll(labels,d,(0,1)) for d in directions]
            best=np.full((h,w),np.inf,np.float32);winner=labels.copy()
            for k,center in enumerate(centers):
                cost=norm+float(np.dot(center,center))-2*(f[...,0]*center[0]+f[...,1]*center[1]+f[...,2]*center[2])
                vote=sum(weight*(neighbor==k) for weight,neighbor in zip(weights,neighbors))
                cost-=strength*vote
                take=cost<best;best[take]=cost[take];winner[take]=k
            take=parity==color_pass;labels[take]=winner[take]
    return labels


def exact_labels(lab,centers):
    pixels=lab.reshape(-1,3).astype(np.float32);result=np.empty(len(pixels),np.uint8)
    for start in range(0,len(pixels),65536):
        _,nearest=cv2.batchDistance(pixels[start:start+65536],centers,cv2.CV_32F,normType=cv2.NORM_L2,K=1)
        result[start:start+len(nearest)]=nearest[:,0]
    return result.reshape(lab.shape[:2])


def clean_islands(labels,centers,features,min_area,passes=2):
    result=labels.copy();h,w=labels.shape
    for k in range(len(centers)):
        n,components,stats,_=cv2.connectedComponentsWithStats((labels==k).astype(np.uint8),8)
        for i in range(1,n):
            x,y,bw,bh,area=stats[i]
            if area>=min_area:continue
            x0=max(0,x-1);y0=max(0,y-1);x1=min(w,x+bw+1);y1=min(h,y+bh+1)
            island=components[y0:y1,x0:x1]==i
            protected=np.max(features[y0:y1,x0:x1][island])>.25
            border=cv2.dilate(island.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~island
            neighbors=result[y0:y1,x0:x1][border];neighbors=neighbors[neighbors!=k]
            if not len(neighbors):continue
            counts=np.bincount(neighbors,minlength=len(centers));candidates=np.flatnonzero(counts)
            distance=np.sum((centers[candidates]-centers[k])**2,axis=1)
            if protected and area>=4 and distance.min()>28**2:continue
            score=distance/(1+np.sqrt(counts[candidates]))
            result[y0:y1,x0:x1][island]=candidates[np.argmin(score)]
    return clean_islands(result,centers,features,min_area,passes-1) if passes>1 else result


def illustration_layers(prepared,p,cancel=None):
    from .core import fit_palette
    from .npr import flow_lines
    blur=prepared['blur'];features=prepared['features'];h,w=blur.shape[:2]
    if cancel and cancel():raise InterruptedError('处理已取消')
    # Mean-shift avoids choosing one large square quadrant over an eye or mouth.
    if min(h,w)>=16:
        scale=min(1.,640/max(h,w))
        small=cv2.resize(blur,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        regions=cv2.pyrMeanShiftFiltering(small,sp=5+p.smooth*1.1,sr=20+p.smooth*2.4,maxLevel=1,
            termcrit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,4,1.))
        regions=cv2.resize(regions,(w,h),interpolation=cv2.INTER_LINEAR)
        detail=cv2.bilateralFilter(blur,5,32,5)
        amount=features[...,None]*.72
        regions=np.clip(regions*(1-amount)+detail*amount,0,255).astype(np.uint8)
    else:regions=blur.copy()
    centers=fit_palette(regions,p.colors,prepared['mask'])
    lab=cv2.cvtColor(regions,cv2.COLOR_RGB2LAB)
    initial=exact_labels(lab,centers)
    labels=spatial_labels(lab,centers,initial,p.smooth,features,cancel)
    labels=clean_islands(labels,centers,features,max(4,round(h*w*(.00006+p.smooth*.000012))))
    palette=cv2.cvtColor(np.clip(centers,0,255).astype(np.uint8)[None],cv2.COLOR_LAB2RGB)[0]
    base=palette[labels]
    # Only strong source structures get ink; palette step boundaries are not ink.
    coarse,internal,tone,_=flow_lines(regions,p.smooth)
    coarse*=.28;internal*=.035
    return {'base':base,'coarse':coarse,'internal':internal,'tone':tone,'strokes':np.zeros_like(tone),
            'flat_labels':labels,'flat_palette':palette,'flat_info':{'palette_size':len(palette),'spatial_passes':4}}
