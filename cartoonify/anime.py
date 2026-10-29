"""Offline portrait generators and scale-normalized head crops; CPU only."""
from pathlib import Path
import hashlib
import json
import threading
import cv2
import numpy as np

MODEL_LIMIT=640
MODES=('动漫增强','柔和插画')
_SESSION=None
_SESSION_NAME=None
_LOCK=threading.RLock()


def session(name='v2'):
    """Retain one model session: changing styles cannot accumulate CPU arenas."""
    global _SESSION,_SESSION_NAME
    files={'v2':('face_paint_v2.onnx','anime_metadata.json'),
           'webtoon':('webtoon.onnx','anime_webtoon_metadata.json')}
    if name not in files:raise ValueError('未知动漫模型')
    with _LOCK:
        if _SESSION is None or _SESSION_NAME!=name:
            import onnxruntime as ort
            root=Path(__file__).resolve().parents[1]/'models'
            filename,record=files[name];path=root/filename
            if not path.exists():raise ValueError('缺少动漫模型，请重新下载完整项目：'+filename)
            metadata=json.loads((root/record).read_text(encoding='utf-8'))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['onnx_sha256']:
                raise ValueError('动漫模型校验失败，请重新下载完整项目。')
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.enable_cpu_mem_arena=False
            _SESSION=None
            _SESSION=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            _SESSION_NAME=name
            if _SESSION.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('动漫模型必须使用 CPU')
        return _SESSION


def _infer(image,name='v2'):
    h,w=image.shape[:2];limit=512 if name=='webtoon' else MODEL_LIMIT
    scale=min(1.,limit/max(h,w))
    small=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else image
    sh,sw=small.shape[:2]
    ph=max(32,((sh+3)//4)*4)-sh;pw=max(32,((sw+3)//4)*4)-sw
    padded=cv2.copyMakeBorder(small,0,ph,0,pw,cv2.BORDER_REFLECT)
    tensor=np.ascontiguousarray(padded.transpose(2,0,1)[None].astype(np.float32)/127.5-1)
    with _LOCK:output=session(name).run(['anime'],{'image':tensor})[0]
    result=np.clip((output[0].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)[:sh,:sw]
    return cv2.resize(result,(w,h),interpolation=cv2.INTER_CUBIC) if result.shape[:2]!=(h,w) else result


def head_transform(box,points=None,size=512,image_shape=None):
    """Eyes determine roll only; preserve individual face proportions and expression."""
    values=np.asarray(box,dtype=np.float64)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.max(np.abs(values))>100000:return None
    x,y,bw,bh=values
    if bw<50 or bh<50 or not .5<=bw/bh<=1.65:return None
    angle=0.
    if points is not None:
        p=np.asarray(points,dtype=np.float64)
        if p.shape==(5,2) and np.isfinite(p).all():
            eyes=p[:2][np.argsort(p[:2,0])];d=eyes[1]-eyes[0]
            candidate=np.degrees(np.arctan2(d[1],d[0]));distance=float(np.linalg.norm(d))
            if .18*bw<distance<.85*bw and abs(candidate)<40:angle=float(candidate)
    side=max(bw*1.95,bh*1.7)
    center=(float(x+bw*.5),float(y+bh*.40))
    if image_shape is not None:
        h,w=image_shape[:2];side=min(side,h,w)
        center=(float(np.clip(center[0],side/2,w-side/2)),float(np.clip(center[1],side/2,h-side/2)))
    transform=cv2.getRotationMatrix2D(center,angle,size/side)
    transform[:,2]+=np.array([size*.5-center[0],size*.5-center[1]])
    return transform


def _head_alpha(size=512):
    # Analytic feather stays exactly zero at crop edges, unlike a blurred ellipse.
    yy,xx=np.mgrid[:size,:size].astype(np.float32)
    radius=np.sqrt(((xx/size-.5)/.45)**2+((yy/size-.49)/.49)**2)
    a=np.clip((1-radius)/.16,0,1);a=a*a*(3-2*a)
    edge=np.minimum.reduce([xx,yy,size-1-xx,size-1-yy])
    return a*np.clip(edge/(size*.035),0,1)


def preserve_skin_tone(source,generated,boxes):
    """Match broad face color statistics only; never restore photographic texture."""
    h,w=source.shape[:2];original=cv2.cvtColor(source.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    drawn=cv2.cvtColor(generated.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    result=drawn.copy();weight=np.zeros((h,w),np.float32);delta=np.zeros_like(drawn)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    for box in boxes:
        if head_transform(box) is None:continue
        x,y,bw,bh=map(float,box)
        # Face-constrained warm chroma gate is a heuristic, not segmentation.
        def skin(rgb):
            r,g,b=cv2.split(rgb.astype(np.float32))
            return (r>g*.98)&(r>b*1.06)&((r-b)>10)
        sample=((xx-(x+bw*.5))/(bw*.34))**2+((yy-(y+bh*.55))/(bh*.38))**2<1
        a=sample&skin(source);b=sample&skin(generated)
        if a.sum()<80 or b.sum()<80:continue
        shift=np.median(original[a],axis=0)-np.median(drawn[b],axis=0)
        shift=np.clip(shift,[-18,-6,-6],[4,6,6])*.7
        # Apply over head/neck with a broad continuous falloff, not a face oval.
        radius=np.sqrt(((xx-(x+bw*.5))/(bw*.95))**2+((yy-(y+bh*.65))/(bh*1.35))**2)
        support=np.clip((1.25-radius)/.6,0,1);support=support*support*(3-2*support)
        support*=skin(generated)
        support=cv2.GaussianBlur(support,(0,0),max(3,bw*.10))
        # Overlapping boxes contribute once; they cannot repeatedly darken a face.
        take=support>weight;delta[take]=shift;weight[take]=support[take]
    result+=delta*weight[...,None]
    return np.clip(cv2.cvtColor(result,cv2.COLOR_LAB2RGB)*255,0,255).astype(np.uint8)


def prepare_head_input(crop):
    """Recover low-contrast eyebrows in bright faces before drawing, not after."""
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)[160:400,144:368]
    low,high=np.percentile(gray,[10,90])
    if np.median(gray)<=185 or not 8<high-low<80:return crop,False
    lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)
    lab[...,0]=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(lab[...,0])
    return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB),True


def stylize(image,face_boxes=(),diagnostics=None,*,mode='动漫增强',landmarks=(),cancel=None):
    if mode not in MODES:raise ValueError('未知动漫表现档位')
    def checkpoint():
        if cancel and cancel():raise InterruptedError('处理已取消')
    checkpoint()
    model='webtoon' if mode=='动漫增强' else 'v2'
    result=_infer(image,model).astype(np.float32);h,w=image.shape[:2];used=aligned=boosted=0;refined=[]
    for box in sorted(face_boxes,key=lambda b:float(b[2])*float(b[3]),reverse=True):
        transform=head_transform(box,image_shape=image.shape)
        if transform is None:continue
        x,y,bw,bh=map(float,box)
        if x+bw<=0 or y+bh<=0 or x>=w or y>=h:continue
        if mode=='柔和插画' and bw*min(1.,MODEL_LIMIT/max(h,w))>=160:continue
        points=next((points for candidate,points in landmarks if np.array_equal(candidate,box)),None)
        transform=head_transform(box,points,image_shape=image.shape)
        checkpoint()
        crop=cv2.warpAffine(image,transform,(512,512),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        if model=='webtoon':
            crop,enhanced=prepare_head_input(crop);boosted+=int(enhanced)
        # Webtoon was distilled at 256: 512 often weakens iris/pupil drawing.
        model_input=cv2.resize(crop,(256,256),interpolation=cv2.INTER_AREA) if model=='webtoon' else crop
        generated=_infer(model_input,model)
        if generated.shape[:2]!=(512,512):generated=cv2.resize(generated,(512,512),interpolation=cv2.INTER_CUBIC)
        inverse=cv2.invertAffineTransform(transform)
        # Warp premultiplied color to avoid dark halos at feathered head boundaries.
        alpha=_head_alpha()
        a=cv2.warpAffine(alpha,inverse,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[...,None]
        color=cv2.warpAffine(generated.astype(np.float32)*alpha[...,None],inverse,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        result=result*(1-a)+color
        used+=1;aligned+=int(points is not None);refined.append(box)
        if used==3:break
    checkpoint()
    if diagnostics is not None:diagnostics.update({'refined_faces':used,'aligned_faces':aligned,'contrast_heads':boosted,'model':model,'mode':mode})
    result=np.clip(result,0,255).astype(np.uint8)
    return preserve_skin_tone(image,result,refined) if model=='webtoon' and used else result
