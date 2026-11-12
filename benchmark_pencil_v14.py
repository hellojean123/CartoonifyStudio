from pathlib import Path
import sys,time,json,threading
import numpy as np,cv2,psutil
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root));out=root/'outputs/pencil_v14'
from cartoonify.core import process,load_image,Params,PRESETS,STYLES,CACHE,resize,_prepare
from cartoonify.pencil import pencil_layers
from cartoonify.strokes import rasterize
source=load_image(root/'samples/35_shu_qi.jpg');params=Params(*PRESETS[3]);rows=[]
for w,h in [(640,480),(1920,1080),(3840,2160)]:
    a=cv2.resize(source,(w,h));CACHE.clear();stop=threading.Event();peak=[psutil.Process().memory_info().rss];base=peak[0]
    def monitor():
        while not stop.wait(.01):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();diag={};start=time.perf_counter()
    try:b=process(a,STYLES[3],params,diagnostics=diag)
    finally:stop.set();watcher.join()
    elapsed=time.perf_counter()-start;assert b.shape==a.shape
    start=time.perf_counter();preview=process(a,STYLES[3],params,preview=True);warm=time.perf_counter()-start
    delta=np.abs(preview.astype(float)-resize(b,1000).astype(float)).mean()
    row={'size':[w,h],'cold_seconds':round(elapsed,3),'warm_preview_seconds':round(warm,3),'peak_rss_mib':round(peak[0]/2**20,1),'baseline_rss_mib':round(base/2**20,1),'preview_export_mae':round(float(delta),3),'stages':diag['stages']};rows.append(row)
    Image.fromarray(b).save(out/f'benchmark_{w}x{h}.png');print(json.dumps(row),flush=True)
(out/'benchmark.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
