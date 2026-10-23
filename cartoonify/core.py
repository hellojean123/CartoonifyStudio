"""Deterministic, tiled CPU stylization. Independent of UI and network; optional bundled CPU face locator."""
from dataclasses import dataclass, replace
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageCms
from .cache import ByteLRU

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)
MAX_PIXELS=16_000_000
ANALYSIS_SIZE=1000
STYLES=['经典卡通','扁平插画','美式漫画','铅笔素描']
PRESETS=[(.5,6,16,1.03),(.24,6,14,1.),(.7,5,8,1.15),(.55,5,16,1.02)]
CACHE=ByteLRU(112*1024**2)
_CV_LOCK=threading.RLock()

@dataclass(frozen=True)
class Params:
    outline: float=.5
    smooth: int=6
    colors: int=16
    contrast: float=1.03
    colored: bool=False
    protect: bool=True
    face_strength: float=.75
    outer: float=.8
    inner: float=.5
    dot_spacing: float=7.
    dot_angle: float=15.
    dot_opacity: float=.28
    pencil_shade: float=.48
    paper: float=.08
    face_boxes: str=''

def normalized(p,style):
    limits={'outline':(0,1),'smooth':(1,10),'colors':(4,48 if style!=STYLES[2] else 12),'contrast':(.6,1.8),
            'face_strength':(0,1),'outer':(0,1),'inner':(0,1),'dot_spacing':(4,18),'dot_angle':(0,90),
            'dot_opacity':(0,.65),'pencil_shade':(0,.8),'paper':(0,.4)}
    updates={}
    for name,(lo,hi) in limits.items():
        value=float(getattr(p,name))
        if not math.isfinite(value): raise ValueError('参数必须是有限数字')
        updates[name]=int(np.clip(value,lo,hi)) if name in ('smooth','colors') else float(np.clip(value,lo,hi))
    return replace(p,**updates)

def load_image(path):
    with Image.open(path) as source:
        if source.width*source.height>MAX_PIXELS or max(source.size)>20000: raise ValueError('图片超过 1600 万像素或单边 20000，请先缩小。')
        im=ImageOps.exif_transpose(source)
        alpha=im.convert('RGBA').getchannel('A') if ('A' in im.getbands() or 'transparency' in im.info) else None
        profile=source.info.get('icc_profile')
        if profile:
            try:
                rgb=im if im.mode in ('RGB','CMYK','LAB') else im.convert('RGB')
                im=ImageCms.profileToProfile(rgb,ImageCms.ImageCmsProfile(BytesIO(profile)),ImageCms.createProfile('sRGB'),outputMode='RGB')
            except (ValueError,OSError,ImageCms.PyCMSError) as exc: raise ValueError('无法读取色彩配置，请先转换为 sRGB') from exc
        else: im=im.convert('RGB')
        if alpha is not None:
            rgba=im.convert('RGBA');rgba.putalpha(alpha)
            im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
        return np.array(im,dtype=np.uint8)

def resize(image,limit):
    h,w=image.shape[:2];scale=min(1.,limit/max(h,w))
    return cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else image.copy()

def fingerprint(im): return hashlib.blake2b(memoryview(np.ascontiguousarray(im)),digest_size=16).hexdigest()

def contrast_image(im,contrast):
    lut=np.clip((np.arange(256,dtype=np.float32)-127.5)*contrast+127.5,0,255).astype(np.uint8)
    return cv2.LUT(im,lut)

def face_mask(image,manual='',details=False):
    h,w=image.shape[:2];mask=np.zeros((h,w),np.float32)
    small=resize(image,640);ratio=w/small.shape[1]
    with _CV_LOCK:
        detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        found=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(24,24))
        # A second cascade and equalized illumination rescue close crops/low contrast.
        alternate=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
        candidates=list(found)+list(alternate.detectMultiScale(cv2.equalizeHist(gray),1.08,4,minSize=(24,24)))
        accepted=[]
        for candidate in sorted(candidates,key=lambda b:int(b[2])*int(b[3]),reverse=True):
            x,y,bw,bh=map(int,candidate)
            if not any(max(0,min(x+bw,a+c)-max(x,a))*max(0,min(y+bh,b+d)-max(y,b)) /
                       max(1,min(bw*bh,c*d))>.45 for a,b,c,d in accepted):
                accepted.append((x,y,bw,bh))
        found=accepted
    landmarks=[]
    model=Path(__file__).resolve().parents[1]/'models/face_detection_yunet_2023mar.onnx'
    if model.exists() and min(h,w)>=32:
        with _CV_LOCK:
            for limit in (320,640):
                detection=resize(image,limit);dh,dw=detection.shape[:2]
                net=cv2.FaceDetectorYN.create(str(model),'',(dw,dh),.7,.3,5000,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
                _,faces=net.detect(cv2.cvtColor(detection,cv2.COLOR_RGB2BGR))
                if faces is not None:
                    for face in faces:
                        box=(face[:4]*[w/dw,h/dh,w/dw,h/dh]).round().astype(int)
                        x,y,bw,bh=box
                        landmarks.append((box,face[4:14].reshape(5,2)*[w/dw,h/dh]))
                    break
    boxes=[(round(x*ratio),round(y*ratio),round(bw*ratio),round(bh*ratio)) for x,y,bw,bh in found]
    # Prefer model boxes/landmarks; avoid a second shifted ellipse on the same face.
    model_boxes=[tuple(map(int,box)) for box,_ in landmarks]
    for x,y,bw,bh in boxes:
        if not any(max(0,min(x+bw,a+c)-max(x,a))*max(0,min(y+bh,b+d)-max(y,b))/max(1,min(bw*bh,c*d))>.35 for a,b,c,d in model_boxes):
            model_boxes.append((x,y,bw,bh))
    boxes=model_boxes
    if manual.strip():
        try:
            entries=json.loads(manual)
            if not isinstance(entries,list) or len(entries)>30: raise ValueError()
            for box in entries:
                if len(box)!=4 or not all(math.isfinite(float(v)) and 0<=float(v)<=1 for v in box): raise ValueError()
                x,y,bw,bh=map(float,box)
                if bw<=0 or bh<=0 or x+bw>1.001 or y+bh>1.001: raise ValueError()
                boxes.append((round(x*w),round(y*h),max(1,round(bw*w)),max(1,round(bh*h))))
        except (TypeError,ValueError,json.JSONDecodeError) as exc: raise ValueError('手动区域格式：[[x,y,宽,高]]，数值为 0～1，区域须位于图内') from exc
    for x,y,bw,bh in boxes: cv2.ellipse(mask,(x+bw//2,y+bh//2),(max(1,round(bw*.54)),max(1,round(bh*.57))),0,0,360,1.,-1)
    mask=cv2.GaussianBlur(mask,(0,0),max(1,min(h,w)*.012))
    result=(np.clip(mask,0,1),boxes)
    return (*result,landmarks) if details else result

def portrait_regions(image,mask,boxes,landmarks=()):
    """Soft anatomical zones; heuristic locations, not semantic segmentation."""
    h,w=mask.shape;features=np.zeros_like(mask)
    for x,y,bw,bh in boxes:
        # Keep eyes, nose and mouth while suppressing cheek/forehead micro-lines.
        for cx,cy,rx,ry in ((.30,.40,.17,.10),(.70,.40,.17,.10),(.50,.60,.11,.17),(.50,.78,.23,.10)):
            cv2.ellipse(features,(round(x+cx*bw),round(y+cy*bh)),
                        (max(1,round(rx*bw)),max(1,round(ry*bh))),0,0,360,1.,-1)
    # Detected landmarks override guessed eye/mouth positions where available.
    for (x,y,bw,bh),points in landmarks:
        x0=max(0,x);y0=max(0,y);x1=min(w,x+bw);y1=min(h,y+bh)
        features[y0:y1,x0:x1]=0
        for index,(px,py) in enumerate(points):
            rx=.15 if index<2 else .10;ry=.085 if index<2 else .12
            cv2.ellipse(features,(round(px),round(py)),(max(1,round(bw*rx)),max(1,round(bh*ry))),0,0,360,1.,-1)
    features=cv2.GaussianBlur(features,(0,0),max(1,min(h,w)*.004))
    rgb=image.astype(np.float32);r,g,b=cv2.split(rgb)
    # Broad chroma gate only inside a detected face; includes dark skin.
    warm=((r>g*.93)&(r>b*1.03)&((r-b)>5)).astype(np.float32)
    skin=mask*cv2.GaussianBlur(warm,(0,0),max(1,min(h,w)*.006))
    return skin,np.clip(features,0,1)*mask

def merge_small_regions(labels,k,min_area=32):
    """Merge isolated palette islands into their dominant adjacent region."""
    result=labels.copy()
    for color in range(k):
        n,components,stats,_=cv2.connectedComponentsWithStats((labels==color).astype(np.uint8),8)
        for component in range(1,n):
            x,y,w,h,area=stats[component]
            if area>=min_area: continue
            x0=max(0,x-1);y0=max(0,y-1);x1=min(labels.shape[1],x+w+1);y1=min(labels.shape[0],y+h+1)
            island=components[y0:y1,x0:x1]==component
            border=cv2.dilate(island.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~island
            neighbors=result[y0:y1,x0:x1][border];neighbors=neighbors[neighbors!=color]
            if neighbors.size: result[y0:y1,x0:x1][island]=np.bincount(neighbors,minlength=k).argmax()
    return result


def fit_palette(im,colors,mask=None):
    pixels=cv2.cvtColor(im,cv2.COLOR_RGB2LAB).reshape(-1,3)
    rng=np.random.default_rng(42);indices=rng.choice(len(pixels),min(6000,len(pixels)),replace=False)
    if mask is not None:
        faces=np.flatnonzero(mask.reshape(-1)>.5)
        if len(faces): indices=np.concatenate([indices,rng.choice(faces,min(2000,len(faces)),replace=False)])
    sample=pixels[indices].astype(np.float32);k=min(colors,len(np.unique(sample,axis=0)))
    if k==1: return sample.mean(axis=0,keepdims=True)
    with _CV_LOCK:
        cv2.setRNGSeed(42)
        _,_,centers=cv2.kmeans(sample,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,.7),1,cv2.KMEANS_PP_CENTERS)
    return centers

def map_palette(im,centers):
    # 32^3 RGB LUT bounds the distance matrix independently of image dimensions.
    axis=(np.arange(32)*8+4).clip(0,255).astype(np.uint8)
    grid=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(128,256,3)
    lab=cv2.cvtColor(grid,cv2.COLOR_RGB2LAB).reshape(-1,3).astype(np.float32)
    _,nearest=cv2.batchDistance(lab,centers,cv2.CV_32F,normType=cv2.NORM_L2,K=1)
    lookup=nearest[:,0].astype(np.uint8)
    index=(im[...,0].astype(np.int32)>>3)*1024+(im[...,1].astype(np.int32)>>3)*32+(im[...,2].astype(np.int32)>>3)
    return lookup[index]

def region_cleanup(labels,k,radius=2):
    best=np.zeros(labels.shape,np.float32);winner=labels.copy();own=np.zeros(labels.shape,np.float32)
    for i in range(k):
        vote=cv2.boxFilter((labels==i).astype(np.float32),-1,(radius*2+1,radius*2+1))
        selected=labels==i;own[selected]=vote[selected]
        take=vote>best;winner[take]=i;best[take]=vote[take]
    result=labels.copy();change=(own<.30)&(best>own+.16);result[change]=winner[change]
    return result

def quantize(im,colors):
    centers=fit_palette(im,int(np.clip(colors,4,48)))
    labels=region_cleanup(map_palette(im,centers),len(centers))
    palette=cv2.cvtColor(np.clip(centers,0,255).astype(np.uint8)[None,:,:],cv2.COLOR_LAB2RGB)[0]
    return palette[labels]

def xdog(gray,smooth):
    f=gray.astype(np.float32)/255;sigma=.65+smooth*.07
    difference=cv2.GaussianBlur(f,(0,0),sigma)-cv2.GaussianBlur(f,(0,0),sigma*1.6)
    return np.clip(-np.tanh((difference+.007)*70),0,1)

def _prepare(image,p,cache=None,digest=None):
    work=resize(image,ANALYSIS_SIZE);guide=contrast_image(work,p.contrast)
    blur=cv2.bilateralFilter(guide,5+2*(p.smooth//3),35+p.smooth*7,10+p.smooth*3)
    if p.protect:
        mask,boxes,landmarks=cache.compute(('faces-v3',digest,p.face_boxes),lambda:face_mask(work,p.face_boxes,True)) if cache is not None else face_mask(work,p.face_boxes,True)
        skin,features=portrait_regions(work,mask,boxes,landmarks)
        amount=skin[...,None]*(1-features[...,None]*.85)*p.face_strength*.32
        blur=np.clip(blur*(1-amount)+cv2.bilateralFilter(blur,7,90,25)*amount,0,255).astype(np.uint8)
    else: mask=np.zeros(work.shape[:2],np.float32);boxes=[];skin=mask;features=mask
    return {'guide':guide,'blur':blur,'mask':mask,'skin':skin,'features':features,'boxes':boxes}

def _layers(prepared,style,p):
    blur=prepared['blur'];mask=prepared['mask'];gray=cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)
    soft=cv2.GaussianBlur(gray,(0,0),1.1)
    coarse=cv2.Canny(soft,75,165).astype(np.float32)/255
    coarse=np.clip(cv2.GaussianBlur(coarse,(0,0),.6)*2.5,0,1)
    internal=xdog(gray,p.smooth)
    if style==STYLES[0]:
        base=cv2.bilateralFilter(blur,7,80,25).astype(np.float32)
        base=quantize(base.astype(np.uint8),20).astype(np.uint8)
        adaptive=cv2.adaptiveThreshold(cv2.medianBlur(gray,5),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,6)
        internal=internal*.55+(1-adaptive.astype(np.float32)/255)*.45
    elif style==STYLES[1]:
        region_input=resize(blur,480)
        region_scale=region_input.shape[1]/blur.shape[1]
        regions=cv2.pyrMeanShiftFiltering(region_input,max(3,round((7+p.smooth)*region_scale)),16+p.smooth,maxLevel=1) if min(blur.shape[:2])>=8 else region_input
        if regions.shape!=blur.shape:
            a,b=guided_coefficients(region_input,regions)
            gray_guide=cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            a=cv2.resize(a,(blur.shape[1],blur.shape[0]));b=cv2.resize(b,(blur.shape[1],blur.shape[0]))
            regions=np.clip((a*gray_guide[...,None]+b)*255,0,255).astype(np.uint8)
        centers=fit_palette(regions,p.colors,mask)
        labels=merge_small_regions(region_cleanup(map_palette(regions,centers),len(centers)),len(centers),max(8,round(regions.shape[0]*regions.shape[1]*.000045)))
        # Restore small facial structures with palette labels, never continuous RGB.
        detail_labels=map_palette(blur,centers)
        labels=np.where(prepared['features']>.25,detail_labels,labels).astype(np.uint8)
        palette=cv2.cvtColor(np.clip(centers,0,255).astype(np.uint8)[None,:,:],cv2.COLOR_LAB2RGB)[0]
        base=palette[labels]
        # Keep a discrete palette: do not blend continuous photographic skin back in.
        internal*=.28;coarse*=.55
    elif style==STYLES[2]:
        comic=cv2.bilateralFilter(cv2.medianBlur(blur,5),7,90,30)
        hsv=cv2.cvtColor(comic,cv2.COLOR_RGB2HSV)
        hsv[...,1]=np.clip(hsv[...,1].astype(np.float32)*1.08,0,255).astype(np.uint8)
        hsv[...,2]=np.clip(np.round(hsv[...,2].astype(np.float32)/255*(p.colors-1))/(p.colors-1)*255,0,255).astype(np.uint8)
        base=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        coarse=cv2.dilate(coarse,np.ones((2,2),np.uint8));internal*=.24
        broad=cv2.GaussianBlur(gray.astype(np.float32)/255,(0,0),3.5)
        shadow=np.clip((.25-broad)/.16,0,1)*(1-prepared['mask']*.65)
        base=np.clip(base.astype(np.float32)*(1-shadow[...,None]*.35),0,255).astype(np.uint8)
    else:
        tone=cv2.GaussianBlur(gray.astype(np.float32)/255,(0,0),2+p.smooth*.5)
        base=np.repeat(np.full_like(tone,250)[...,None],3,axis=2)
        if p.colored: base*=.58+.42*blur.astype(np.float32)/255
        base=np.clip(base,0,255).astype(np.uint8);coarse*=1.1;internal=np.clip(internal*1.1,0,1)
    return {'direction':cv2.GaussianBlur(cv2.Sobel(gray.astype(np.float32),cv2.CV_32F,1,0),(0,0),4), 'base':base,'coarse':coarse,'internal':internal,'tone':cv2.GaussianBlur(gray.astype(np.float32)/255,(0,0),2.)}

def guided_coefficients(guide,base):
    i=cv2.cvtColor(guide,cv2.COLOR_RGB2GRAY).astype(np.float32)/255;p=base.astype(np.float32)/255
    mi=cv2.boxFilter(i,-1,(11,11));mp=cv2.boxFilter(p,-1,(11,11))
    variance=cv2.boxFilter(i*i,-1,(11,11))-mi*mi
    covariance=cv2.boxFilter(i[...,None]*p,-1,(11,11))-mi[...,None]*mp
    a=np.clip(covariance/(variance[...,None]+.0015),-1.5,1.5);b=mp-a*mi[...,None]
    return cv2.boxFilter(a,-1,(11,11)),cv2.boxFilter(b,-1,(11,11))

def sample_region(low,target_shape,bounds):
    y0,y1,x0,x1=bounds;h,w=target_shape
    xs=(np.arange(x0,x1,dtype=np.float32)+.5)*(low.shape[1]/w)-.5
    ys=(np.arange(y0,y1,dtype=np.float32)+.5)*(low.shape[0]/h)-.5
    mx,my=np.meshgrid(xs,ys)
    return cv2.remap(low,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def render(image,prepared,layers,style,p,quality='精细',output_limit=None,tile_size=512,cancel=None):
    target=resize(image,output_limit) if output_limit else image
    h,w=target.shape[:2];scale=max(h,w)/ANALYSIS_SIZE;result=np.empty((h,w,3),np.uint8)
    coefficients=guided_coefficients(prepared['guide'],layers['base']) if quality=='精细' and max(h,w)>ANALYSIS_SIZE and style==STYLES[0] else None
    halo=max(8,math.ceil(6*max(.65,scale)))
    for y in range(0,h,tile_size):
        if cancel is not None and cancel(): raise InterruptedError('处理已取消')
        for x in range(0,w,tile_size):
            ey0=max(0,y-halo);ey1=min(h,y+tile_size+halo);ex0=max(0,x-halo);ex1=min(w,x+tile_size+halo);bounds=(ey0,ey1,ex0,ex1)
            lh,lw=layers['base'].shape[:2]
            xs=(np.arange(ex0,ex1,dtype=np.float32)+.5)*(lw/w)-.5
            ys=(np.arange(ey0,ey1,dtype=np.float32)+.5)*(lh/h)-.5
            mx,my=np.meshgrid(xs,ys)
            def take(layer): return cv2.remap(layer,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            base=take(layers['base']).astype(np.float32)
            if quality=='精细' and max(h,w)>ANALYSIS_SIZE and style==STYLES[0]:
                original=contrast_image(target[ey0:ey1,ex0:ex1],p.contrast)
                gray=cv2.cvtColor(original,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            if coefficients is not None:
                a=take(coefficients[0]);b=take(coefficients[1])
                base=base*.8+np.clip((a*gray[...,None]+b)*255,0,255)*.2
            mask=take(prepared['mask'])
            outer=take(layers['coarse']);inner=take(layers['internal'])
            if quality=='精细' and max(h,w)>ANALYSIS_SIZE and style==STYLES[0]:
                detail=cv2.GaussianBlur(gray,(0,0),max(.65,scale*.55))
                grad=cv2.magnitude(cv2.Sobel(detail,cv2.CV_32F,1,0),cv2.Sobel(detail,cv2.CV_32F,0,1))
                extra=np.clip((grad-.06/max(1,scale))*1.4*max(1,scale),0,1)
                inner=np.maximum(inner,extra*.24)
            skin=take(prepared['skin']);features=take(prepared['features'])
            suppression=skin*(1-features*.92)
            inner=inner*(1-suppression*p.face_strength*.90)
            outer=outer*(1-suppression*p.face_strength*.55)
            ink=np.clip(p.outline*(p.outer*outer*.88+p.inner*inner*.72),0,.90)
            yy=(np.arange(ey0,ey1,dtype=np.float32)[:,None]+.5)/max(scale,.001)
            xx=(np.arange(ex0,ex1,dtype=np.float32)[None,:]+.5)/max(scale,.001)
            if style in (STYLES[2],STYLES[3]): tone=take(layers['tone'])
            if style==STYLES[2]:
                angle=np.float32(np.deg2rad(p.dot_angle));u=(xx*np.cos(angle)+yy*np.sin(angle))/p.dot_spacing;v=(-xx*np.sin(angle)+yy*np.cos(angle))/p.dot_spacing
                distance=np.sqrt(((u+.5)%1-.5)**2+((v+.5)%1-.5)**2)
                radius=np.sqrt(np.clip((.80-tone)/.65,0,1))*.35
                aa=min(.18,.55/(p.dot_spacing*max(scale,.25)))
                dots=np.clip((radius-distance)/aa+.5,0,1)
                midtone=np.clip((tone-.12)/.2,0,1)*np.clip((.86-tone)/.25,0,1)
                base*=1-dots[...,None]*p.dot_opacity*midtone[...,None]*(1-mask[...,None]*p.face_strength*.8)
            if style==STYLES[3]:
                # Continuous graphite tone carries form even with no paper/strokes.
                density=np.clip((.98-tone)/.98,0,1)**.85
                base*=1-(density*p.pencil_shade*1.42)[...,None]
                # Low-amplitude irregular grain replaces screen-like global crosshatching.
                grain=(np.sin(xx*2.79+yy*1.31)*np.sin(xx*.67-yy*2.13))
                base-=grain[...,None]*(p.paper*13+density[...,None]*p.pencil_shade*5)
                flow=np.tanh(take(layers['direction'])/28.)
                stroke=np.sin(xx*1.9+yy*.8+flow*2.4)*np.sin(xx*.11+yy*.17)
                base-=stroke[...,None]*density[...,None]*p.pencil_shade*18
            tile=np.clip(base*(1-ink[...,None]),0,255).astype(np.uint8);ch=min(tile_size,h-y);cw=min(tile_size,w-x)
            result[y:y+ch,x:x+cw]=tile[y-ey0:y-ey0+ch,x-ex0:x-ex0+cw]
    return result

def process(image,style,params=None,preview=False,quality='精细',cache=CACHE,preview_limit=1000,tile_size=512,cancel=None,diagnostics=None):
    if style not in STYLES: raise ValueError('未知风格')
    if quality not in ('标准','精细'): raise ValueError('未知质量档位')
    p=normalized(params or Params(*PRESETS[STYLES.index(style)]),style);im=np.asarray(image)
    if im.ndim==2 and im.size: im=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    if im.ndim!=3 or im.shape[2]!=3 or im.size==0 or im.dtype!=np.uint8: raise ValueError('需要非空 uint8 RGB 图像')
    if im.shape[0]*im.shape[1]>MAX_PIXELS or max(im.shape[:2])>20000: raise ValueError('图片超过处理上限')
    started=time.perf_counter()
    digest=fingerprint(im);prekey=('prepare',digest,p.smooth,p.contrast,p.protect,p.face_strength,p.face_boxes)
    prepared=cache.compute(prekey,lambda:_prepare(im,p,cache,digest)) if cache is not None else _prepare(im,p)
    prepared_at=time.perf_counter()
    key=('layers',prekey,style,p.colors if style in STYLES[1:3] else None,p.colored if style==STYLES[3] else None)
    layers=cache.compute(key,lambda:_layers(prepared,style,p)) if cache is not None else _layers(prepared,style,p)
    layers_at=time.perf_counter()
    result=render(im,prepared,layers,style,p,quality,preview_limit if preview else None,tile_size,cancel)
    if diagnostics is not None: diagnostics.update({'faces':prepared['boxes'],'analysis_size':prepared['guide'].shape[:2],'cache':cache.stats() if cache else {},'stages':{'prepare':prepared_at-started,'layers':layers_at-prepared_at,'render':time.perf_counter()-layers_at,'total':time.perf_counter()-started}})
    return result
