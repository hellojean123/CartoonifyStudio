"""Bundled FacePaint v2 inference, explicitly CPU only; no runtime download."""
from pathlib import Path
import hashlib
import json
import threading
import cv2
import numpy as np

MODEL_LIMIT=640
_SESSION=None
_LOCK=threading.RLock()

def session():
    global _SESSION
    with _LOCK:
        if _SESSION is None:
            import onnxruntime as ort
            root=Path(__file__).resolve().parents[1]/'models'
            path=root/'face_paint_v2.onnx'
            if not path.exists(): raise ValueError('缺少动漫模型，请重新下载完整项目（models/face_paint_v2.onnx）。')
            metadata=json.loads((root/'anime_metadata.json').read_text(encoding='utf-8'))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['onnx_sha256']:
                raise ValueError('动漫模型校验失败，请重新下载完整项目。')
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_cpu_mem_arena=False
            _SESSION=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            if _SESSION.get_providers()!=['CPUExecutionProvider']: raise RuntimeError('动漫模型必须使用 CPU')
        return _SESSION

def _infer(image):
    h,w=image.shape[:2];scale=min(1.,MODEL_LIMIT/max(h,w))
    small=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else image
    sh,sw=small.shape[:2]
    # Reflection layers require a minimum spatial size. Pad instead of distorting aspect.
    ph=max(32,((sh+3)//4)*4)-sh;pw=max(32,((sw+3)//4)*4)-sw
    padded=cv2.copyMakeBorder(small,0,ph,0,pw,cv2.BORDER_REFLECT)
    tensor=np.ascontiguousarray(padded.transpose(2,0,1)[None].astype(np.float32)/127.5-1)
    with _LOCK: output=session().run(['anime'],{'image':tensor})[0]
    result=np.clip((output[0].transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)[:sh,:sw]
    return cv2.resize(result,(w,h),interpolation=cv2.INTER_CUBIC) if result.shape[:2]!=(h,w) else result

def stylize(image,face_boxes=(),diagnostics=None):
    result=_infer(image).astype(np.float32);h,w=image.shape[:2];used=0
    for box in sorted(face_boxes,key=lambda b:int(b[2])*int(b[3]),reverse=True):
        x,y,bw,bh=map(int,box)
        # Only resolvable source faces that became small in the global pass.
        if bw<50 or bh<50 or bw*min(1.,MODEL_LIMIT/max(h,w))>=160:continue
        side=min(h,w,round(max(bw*2.15,bh*1.85)))
        if side<32:continue
        cx=x+bw*.5;cy=y+bh*.45
        left=int(np.clip(round(cx-side/2),0,w-side));top=int(np.clip(round(cy-side/2),0,h-side))
        crop=image[top:top+side,left:left+side]
        generated=_infer(cv2.resize(crop,(512,512),interpolation=cv2.INTER_CUBIC))
        generated=cv2.resize(generated,(side,side),interpolation=cv2.INTER_AREA)
        alpha=np.zeros((side,side),np.float32)
        cv2.ellipse(alpha,(round(cx-left),round(cy-top)),(max(1,round(bw*.8)),max(1,round(bh*.85))),0,0,360,1.,-1)
        alpha=cv2.GaussianBlur(alpha,(0,0),max(2,bw*.1))[...,None]
        result[top:top+side,left:left+side]=result[top:top+side,left:left+side]*(1-alpha)+generated*alpha
        used+=1
        if used==3:break
    if diagnostics is not None:diagnostics['refined_faces']=used
    return np.clip(result,0,255).astype(np.uint8)
