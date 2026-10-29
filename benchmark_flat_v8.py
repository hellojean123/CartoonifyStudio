from pathlib import Path
from dataclasses import replace
import argparse,json,subprocess,sys,time,threading
import cv2,psutil
from cartoonify.core import process,load_image,Params,PRESETS,STYLES,CACHE
from cartoonify.service import encode
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/flat_v8'


def one(size):
    source=load_image(ROOT/'samples/23_portrait.jpg')
    image=cv2.resize(source,(round(size*source.shape[1]/source.shape[0]),size),interpolation=cv2.INTER_AREA)
    proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();runs=[];p=Params(*PRESETS[1])
    try:
        for _ in range(2):
            t=time.perf_counter();process(image,STYLES[1],p,cache=None);runs.append(time.perf_counter()-t)
        CACHE.clear();process(image,STYLES[1],p,preview=True)
        before=CACHE.stats();t=time.perf_counter();process(image,STYLES[1],replace(p,outline=.4),preview=True);cached=time.perf_counter()-t
        assert CACHE.stats()['misses']==before['misses']
    finally:stop.set();thread.join()
    return {'shape':list(image.shape),'uncached_seconds':runs,'cached_outline_preview_seconds':cached,'peak_process_rss_mib':peak[0]/1024**2,'rss_sampling_ms':10}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--size',type=int);args=parser.parse_args()
    if args.size:print(json.dumps(one(args.size)),flush=True);return
    rows=[]
    for size in (480,1000,1920,3840):
        r=subprocess.run([sys.executable,__file__,'--size',str(size)],capture_output=True,check=True)
        row=json.loads(r.stdout.decode().splitlines()[-1]);rows.append(row)
        (OUT/'benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    im=load_image(ROOT/'samples/high_resolution/reference_4k.jpg');t=time.perf_counter();result=process(im,STYLES[1]);seconds=time.perf_counter()-t
    assert result.shape==im.shape
    (OUT/'native_4k.png').write_bytes(encode(result,'PNG'));(ROOT/'outputs/highres_2.png').write_bytes(encode(result,'PNG'))
    (OUT/'native_4k.json').write_text(json.dumps({'shape':list(im.shape),'algorithm_seconds':seconds,'source':'samples/high_resolution/reference_4k.jpg'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
