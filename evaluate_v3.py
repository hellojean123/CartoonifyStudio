"""Reproducible full dataset and native crop comparison against committed v2."""
from pathlib import Path
import importlib.util
import subprocess
import json
import time
import html
import cv2
import numpy as np
from PIL import Image
from cartoonify.core import process,load_image,resize,STYLES,CACHE
from evaluate import sheet
from build_gallery import main as build_gallery

ROOT=Path(__file__).parent

def main():
    # Load the exact published baseline in memory without leaving backup copies.
    code=subprocess.check_output(['git','show','2f823627e5fc24ffe0c9b33d478ae136ac226723:cartoonify/core.py'],cwd=ROOT).decode('utf-8')
    import types,sys
    old=types.ModuleType('cartoonify.baseline_v2');old.__package__='cartoonify'
    sys.modules[old.__name__]=old;exec(compile(code,'baseline_v2','exec'),old.__dict__)
    out=ROOT/'outputs';review=out/'v3_review';review.mkdir(exist_ok=True)
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    rows=[];cards=[]
    selected={'01_portrait','03_portrait','05_portrait','13_portrait','15_landscape','17_architecture','19_animal'}
    for record in records:
        path=ROOT/'samples'/record['filename'];im=load_image(path);panels=[im]
        for i,style in enumerate(STYLES):
            CACHE.clear();info={};t=time.perf_counter();result=process(im,style,diagnostics=info);seconds=time.perf_counter()-t
            dest=out/f'{i+1}_{style}';dest.mkdir(exist_ok=True)
            Image.fromarray(result).save(dest/(path.stem+'.png'),compress_level=3)
            preview=process(im,style,preview=True,quality='标准')
            rows.append({'file':path.name,'style':style,'seconds':seconds,'faces':info['faces'],'stages':info['stages'],'preview_export_mae':float(np.abs(preview.astype(float)-resize(result,1000)).mean())})
            panels.append(result)
            if path.stem in selected:
                before=old.process(im,style)
                name=f'{path.stem}_{i+1}.jpg'
                sheet([im,before,result],['Original','v2','v3'],review/name,height=560)
                cards.append(f'<h2>{html.escape(path.stem)} / {style}</h2><img loading="lazy" src="{name}">')
                # Native face/detail crop, independent of thumbnail aesthetics.
                h,w=im.shape[:2];size=min(420,h,w);x=(w-size)//2;y=max(0,(h-size)//3)
                sheet([a[y:y+size,x:x+size] for a in (im,before,result)],['Original crop','v2 crop','v3 crop'],review/(path.stem+f'_{i+1}_crop.jpg'),width=size*3+36,height=size+45)
        sheet(panels,['Original','Classic','Flat','Comic','Pencil'],out/(path.stem+'_comparison.jpg'))
        print('Rendered',path.name,flush=True)
    (review/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>v2 / v3 实际效果</title><style>body{font:16px system-ui;background:#171322;color:#eee;margin:24px}img{max-width:100%}</style><h1>原图 / 已发布 v2 / 优化 v3</h1><p>本页没有自动审美评分；原尺寸局部见同目录 crop 图片。</p>'+''.join(cards),encoding='utf-8')
    (out/'timings_v3.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    high=load_image(ROOT/'samples/high_resolution/reference_4k.jpg');bench=[]
    for i,style in enumerate(STYLES):
        measurements=[]
        for _ in range(3):
            CACHE.clear();t=time.perf_counter();result=process(high,style);measurements.append(time.perf_counter()-t)
        Image.fromarray(result).save(out/f'highres_{i+1}.png',compress_level=3)
        bench.append({'style':style,'shape':list(high.shape),'seconds':measurements,'median_seconds':float(np.median(measurements))})
    (out/'benchmark_v3.json').write_text(json.dumps(bench,ensure_ascii=False,indent=2),encoding='utf-8')
    build_gallery()
    print('84 outputs, 21 comparisons, 28 v2/v3 comparisons + native crops; 4K benchmark complete',flush=True)

if __name__=='__main__':main()
