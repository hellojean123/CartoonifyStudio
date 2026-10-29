"""Published/default/fixed-budget illustration comparisons on all licensed samples."""
from pathlib import Path
from dataclasses import replace
from io import BytesIO
import json,html,hashlib,subprocess,time,threading
import cv2,numpy as np,psutil
from PIL import Image
from cartoonify.core import *
from cartoonify.core import _prepare,_layers
from cartoonify.npr import region_smooth
from evaluate import sheet
from build_gallery import main as build_gallery
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/flat_v8';BASE='f39049b9dffd5ade8f2bbc7abff58cdbdf249b83'


def fragments(labels,k):
    small=count=0
    for i in range(k):
        _,_,stats,_=cv2.connectedComponentsWithStats((labels==i).astype(np.uint8),8)
        areas=stats[1:,cv2.CC_STAT_AREA];selected=areas[areas<32];small+=int(selected.sum());count+=len(selected)
    return {'small_component_pixels':small,'small_component_count':count}


def main():
    OUT.mkdir(exist_ok=True);rows=[];cards=[]
    other={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for i in (0,2,3) for p in (ROOT/f'outputs/{i+1}_{STYLES[i]}').glob('*.png')}
    samples=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images'];proc=psutil.Process()
    for record in samples:
        path=ROOT/'samples'/record['filename'];stem=path.stem;im=load_image(path)
        old=np.array(Image.open(BytesIO(subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASE}:outputs/2_扁平插画/{stem}.png']))).convert('RGB'))
        fixed=Params(.3,7,14,1.03);prepared=_prepare(im,fixed)
        regions=region_smooth(prepared['blur'],fixed.smooth,radius=3+fixed.smooth*2)
        centers=fit_palette(regions,14,prepared['mask'])
        oldlabels=merge_small_regions(region_cleanup(map_palette(regions,centers),len(centers)),len(centers),max(8,round(regions.shape[0]*regions.shape[1]*.00008)))
        oldlabels=np.where(prepared['features']>.25,map_palette(prepared['blur'],centers),oldlabels).astype(np.uint8)
        t=time.perf_counter();fixed_layers=_layers(prepared,STYLES[1],fixed);same_budget=render(im,prepared,fixed_layers,STYLES[1],fixed,'精细')
        fixed_seconds=time.perf_counter()-t
        Image.fromarray(same_budget).save(OUT/f'{stem}_14colors.png',compress_level=3)
        CACHE.clear();info={};peak=[proc.memory_info().rss];stop=threading.Event()
        def monitor():
            while not stop.wait(.01):peak[0]=max(peak[0],proc.memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True);thread.start();t=time.perf_counter()
        try:result=process(im,STYLES[1],diagnostics=info)
        finally:stop.set();thread.join()
        seconds=time.perf_counter()-t
        for dest in (OUT/f'{stem}.png',ROOT/f'outputs/2_扁平插画/{stem}.png'):Image.fromarray(result).save(dest,compress_level=3)
        t=time.perf_counter();preview=process(im,STYLES[1],preview=True,quality='标准');cached=time.perf_counter()-t
        mae=float(np.abs(preview.astype(float)-resize(result,1000).astype(float)).mean());assert mae<4
        sheet([im,old,same_budget,result],['Original','Published 14 colors','New same 14 colors','New default 20 colors'],OUT/f'{stem}_review.jpg',width=1800,height=620)
        if prepared['boxes']:
            x,y,bw,bh=max(prepared['boxes'],key=lambda b:b[2]*b[3]);scale=max(im.shape[:2])/1000 if max(im.shape[:2])>1000 else 1
            x0=max(0,round((x-bw*.15)*scale));x1=min(im.shape[1],round((x+bw*1.15)*scale));y0=max(0,round((y-bh*.15)*scale));y1=min(im.shape[0],round((y+bh*1.15)*scale))
            if x1>x0 and y1>y0:sheet([a[y0:y1,x0:x1] for a in (im,old,same_budget,result)],['Original','Old 14','New 14','New 20'],OUT/f'{stem}_crop.png',width=1600,height=450)
        panels=[im]+[result if i==1 else load_image(ROOT/f'outputs/{i+1}_{STYLES[i]}/{stem}.png') for i in range(4)]
        sheet(panels,['Original','Anime v7','Flat v8','Comic','Pencil v6'],ROOT/f'outputs/{stem}_comparison.jpg',height=420)
        row={'file':path.name,'seconds':seconds,'fixed14_seconds':fixed_seconds,'cached_preview_seconds':cached,'peak_process_rss_mib':peak[0]/1024**2,'preview_export_mae':mae,'old14_fragments':fragments(oldlabels,len(centers)),'new14_fragments':fragments(fixed_layers['flat_labels'],len(fixed_layers['flat_palette'])),'flat':info['flat'],'sha256':hashlib.sha256((OUT/f'{stem}.png').read_bytes()).hexdigest()}
        rows.append(row);(OUT/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        cards.append(f'<article><h2>{html.escape(path.name)}</h2><img loading="lazy" src="{stem}_review.jpg"><p>{seconds:.2f} 秒 · <a href="{stem}.png">新版 PNG</a> · <a href="{stem}_14colors.png">同预算 14 色</a></p><details><summary>照片署名</summary>{html.escape(record["author"])} · {html.escape(record["license"])} · <a href="{html.escape(record["source_url"],quote=True)}">来源</a></details></article>')
        print(json.dumps({'sample':path.name,'seconds':round(seconds,2),'old_small':row['old14_fragments']['small_component_count'],'new_small':row['new14_fragments']['small_component_count']}),flush=True)
    assert len(other)==63 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in other.items())
    (OUT/'unchanged_other_styles.json').write_text(json.dumps(other,ensure_ascii=False,indent=2),encoding='utf-8')
    doc='<!doctype html><meta charset="utf-8"><title>扁平插画 v8</title><style>body{background:#171322;color:#eee;font:16px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}article{margin:30px 0}a{color:#c8adff}</style><h1>扁平插画：原图 / 旧版 14 色 / 新版同预算 / 新版默认 20 色</h1><p>保留人物特征的同时简化区域。边缘抗锯齿颜色不计入主体色板预算。指标只描述碎块，不代表审美评分。</p><a href="../../flat_v8_report.md">实测与已知局限</a>'+''.join(cards)
    (OUT/'gallery.html').write_text(doc,encoding='utf-8');build_gallery()


if __name__=='__main__':main()
