"""Separate-process RSS benchmark; CPU model and render memory are both included."""
from pathlib import Path
from dataclasses import replace
import argparse,json,subprocess,sys,time,threading
import cv2,numpy as np,psutil
from cartoonify.core import process,load_image,resize,Params,PRESETS,STYLES,CACHE
ROOT=Path(__file__).resolve().parent


def one(size,mode):
    source=load_image(ROOT/'samples/23_portrait.jpg')
    image=cv2.resize(source,(round(size*source.shape[1]/source.shape[0]),size),interpolation=cv2.INTER_AREA)
    proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();runs=[];p=replace(Params(*PRESETS[0]),anime_mode=mode)
    try:
        for _ in range(2):
            t=time.perf_counter();result=process(image,STYLES[0],p,cache=None);runs.append(time.perf_counter()-t)
        CACHE.clear();process(image,STYLES[0],p,preview=True)
        before=CACHE.stats();t=time.perf_counter()
        process(image,STYLES[0],replace(p,outline=.6),preview=True)
        cached=time.perf_counter()-t;after=CACHE.stats();assert before['misses']==after['misses']
        if size==3840:
            from PIL import Image
            Image.fromarray(result).save(ROOT/'outputs/anime_v7/highres_portrait.png',compress_level=3)
    finally:stop.set();thread.join()
    return {'shape':list(image.shape),'mode':mode,'uncached_seconds':runs,'cached_outline_preview_seconds':cached,'peak_process_rss_mib':peak[0]/1024**2,'rss_sampling_ms':10,'threads':cv2.getNumThreads(),'opencl':cv2.ocl.useOpenCL()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int);parser.add_argument('--mode',default='动漫增强');args=parser.parse_args()
    if args.size:print(json.dumps(one(args.size,args.mode)),flush=True);return
    rows=[]
    for mode,size in [('动漫增强',480),('动漫增强',1000),('动漫增强',1920),('动漫增强',3840),('柔和插画',1000)]:
        r=subprocess.run([sys.executable,__file__,'--size',str(size),'--mode',mode],capture_output=True,check=True)
        row=json.loads(r.stdout.decode().splitlines()[-1]);rows.append(row)
        (ROOT/'outputs/anime_v7/benchmark.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
