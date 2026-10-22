"""Local jobs, decoded-image cache and export; usable without Gradio."""
from contextlib import contextmanager
from pathlib import Path
from io import BytesIO
import threading
import time
import uuid
import zipfile
from PIL import Image, ImageCms
from .cache import ByteLRU
from .core import load_image, process, resize

ROOT=Path(__file__).resolve().parents[1]
EXPORTS=ROOT/'outputs'/'exports'
DECODE_CACHE=ByteLRU(64*1024**2)
CPU_GATE=threading.Lock()
_JOBS={}
_JOBS_LOCK=threading.Lock()
SRGB=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()

def load_cached(path):
    p=Path(path).resolve();stat=p.stat()
    if stat.st_size>40*1024**2: raise ValueError('文件超过 40 MB')
    key=(str(p),stat.st_mtime_ns,stat.st_size)
    return DECODE_CACHE.compute(key,lambda:load_image(p))

@contextmanager
def job(session):
    with CPU_GATE:
        token=threading.Event()
        with _JOBS_LOCK:
            if len(_JOBS)>200:
                for key in list(_JOBS)[:100]: _JOBS.pop(key,None)
            _JOBS[session]=token
        try: yield token
        finally:
            with _JOBS_LOCK:
                if _JOBS.get(session) is token: _JOBS.pop(session,None)

def cancel_job(session):
    with _JOBS_LOCK:
        token=_JOBS.get(session)
        if token: token.set();return '已请求取消，将在下一个图块或图片边界停止。'
    return '没有正在运行的任务。'

def encode(image,fmt):
    fmt=fmt.upper()
    if fmt not in ('PNG','JPG'): raise ValueError('导出格式只支持 PNG / JPG')
    buffer=BytesIO();im=Image.fromarray(image)
    if fmt=='JPG': im.save(buffer,format='JPEG',quality=95,subsampling=0,icc_profile=SRGB)
    else: im.save(buffer,format='PNG',compress_level=3,icc_profile=SRGB)
    return buffer.getvalue()

def export_one(path,style,params,fmt='PNG',quality='精细',cancel=None):
    image=load_cached(path)
    result=process(image,style,params,quality=quality,cancel=cancel)
    data=encode(result,fmt)
    if cancel and cancel(): raise InterruptedError('处理已取消')
    EXPORTS.mkdir(parents=True,exist_ok=True)
    dest=EXPORTS/f'{uuid.uuid4().hex}.{fmt.lower()}'
    # Atomic publication prevents a cancelled/failed job offering a truncated export.
    tmp=dest.with_suffix('.part');tmp.write_bytes(data);tmp.replace(dest)
    return result,dest

def batch_export(paths,style,params,fmt='PNG',quality='精细',session='batch',progress=None):
    if not paths: raise ValueError('请选择批量照片')
    EXPORTS.mkdir(parents=True,exist_ok=True)
    dest=EXPORTS/f'batch-{uuid.uuid4().hex}.zip';tmp=dest.with_suffix('.part')
    failures=[];success=0;cancelled=False
    with job(session) as token:
        try:
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as z:
                for index,path in enumerate(paths):
                    if token.is_set(): cancelled=True;break
                    if progress: progress(index/len(paths),desc=f'{index+1}/{len(paths)}')
                    try:
                        im=load_cached(path);result=process(im,style,params,quality=quality,cancel=token.is_set)
                        data=encode(result,fmt)
                        if token.is_set(): cancelled=True;break
                        z.writestr(f'{index+1:03d}_{Path(path).stem}.{fmt.lower()}',data);success+=1
                    except InterruptedError: cancelled=True;break
                    except Exception as exc: failures.append(f'{Path(path).name}: {exc}')
                summary=f'成功 {success} / {len(paths)}；失败 {len(failures)}；取消 {cancelled}'
                z.writestr('处理记录.txt',summary+'\n'+'\n'.join(failures))
            tmp.replace(dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    if progress: progress(1,desc=summary)
    return str(dest),summary
