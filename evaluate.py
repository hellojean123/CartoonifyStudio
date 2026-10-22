"""Full-dataset rendering, scale consistency, before/after and blinded comparison."""
from pathlib import Path
import html
import json
import time
import random
import cv2
import numpy as np
from PIL import Image,ImageDraw
from cartoonify.core import STYLES,Params,PRESETS,load_image,process,resize,CACHE
from build_gallery import main as build_gallery
from benchmarks import legacy_v1 as legacy

ROOT=Path(__file__).parent

def sheet(panels,labels,path,width=1500,height=380):
    image=Image.new('RGB',(width,height),'#f3f0fa');draw=ImageDraw.Draw(image);column=width//len(panels)
    for i,(panel,label) in enumerate(zip(panels,labels)):
        thumb=Image.fromarray(panel);thumb.thumbnail((column-12,height-40))
        image.paste(thumb,(i*column+(column-thumb.width)//2,34));draw.text((i*column+8,10),label,fill='black')
    image.save(path,quality=93)

def edge_fragments(image):
    small=resize(image,500)
    edges=cv2.Canny(cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),80,160)
    _,_,stats,_=cv2.connectedComponentsWithStats(edges,8)
    areas=stats[1:,cv2.CC_STAT_AREA]
    return float(areas[areas<12].sum()/max(1,np.count_nonzero(edges)))

def main():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    comparisons=out/'upgrade_comparisons';comparisons.mkdir(exist_ok=True)
    blind=out/'blind';blind.mkdir(exist_ok=True)
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    timings=[];qa=[];blind_cards=[];blind_key=[];rng=random.Random(20261022)
    selected={'01_portrait','03_portrait','05_portrait','13_portrait','15_landscape','19_animal'}
    for record in records:
        path=ROOT/'samples'/record['filename'];original=load_image(path);panels=[original]
        for index,style in enumerate(STYLES):
            info={};CACHE.clear();started=time.perf_counter()
            result=process(original,style,diagnostics=info)
            elapsed=time.perf_counter()-started
            dest=out/f'{index+1}_{style}';dest.mkdir(exist_ok=True)
            t=time.perf_counter();Image.fromarray(result).save(dest/(path.stem+'.png'),compress_level=3);encoding=time.perf_counter()-t
            preview=process(original,style,preview=True,quality='标准')
            down=resize(result,1000)
            difference=np.abs(preview.astype(np.float32)-down.astype(np.float32))
            old=legacy.process(original,style)
            row={'file':path.name,'style':style,'seconds':elapsed,'png_encoding_seconds':encoding,'stages':info['stages'],'faces':info['faces'],'preview_export_mae_0_255':float(difference.mean()),'preview_export_p95_abs_error':float(np.percentile(difference,95)),'fragmented_edge_fraction_before':edge_fragments(old),'fragmented_edge_fraction_after':edge_fragments(result)}
            timings.append(row);panels.append(result)
            if path.stem in selected:
                filename=f'{path.stem}_{index+1}.jpg'
                sheet([original,old,result],['Original','Before v1','After v2'],comparisons/filename)
                # A/B assignment reproducible and hidden from the visible gallery.
                new_first=rng.choice([True,False]);a,b=(result,old) if new_first else (old,result)
                sheet([original,a,b],['Original','A','B'],blind/filename)
                ident=f'{path.stem}_{index+1}'
                blind_cards.append(f'<article><h2>{html.escape(ident)}</h2><img loading="lazy" src="blind/{filename}"><p>选择更自然且更符合该风格的一项：<button onclick="vote(\'{ident}\',\'A\')">A</button> <button onclick="vote(\'{ident}\',\'B\')">B</button> <button onclick="vote(\'{ident}\',\'tie\')">相当</button> <span id="{ident}"></span></p></article>')
                blind_key.append({'id':ident,'A':'v2' if new_first else 'v1','B':'v1' if new_first else 'v2'})
            qa.append({'file':path.name,'style':style,'original_dimensions':list(original.shape[:2]),'output_dimensions':list(result.shape[:2])})
        sheet(panels,['Original','Classic','Flat','Comic','Pencil'],out/(path.stem+'_comparison.jpg'))
        print('Evaluated',path.name,flush=True)
    (out/'timings.json').write_text(json.dumps(timings,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'quality_metrics.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'blind_key.json').write_text(json.dumps(blind_key,ensure_ascii=False,indent=2),encoding='utf-8')
    script="""<script>let votes=JSON.parse(localStorage.getItem('cartoonify-v2-votes')||'{}');function vote(k,v){votes[k]=v;localStorage.setItem('cartoonify-v2-votes',JSON.stringify(votes));document.getElementById(k).textContent='已选 '+v}function download(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(votes,null,2)],{type:'application/json'}));a.download='blind_votes.json';a.click()}for(let[k,v]of Object.entries(votes)){let e=document.getElementById(k);if(e)e.textContent='已选 '+v}</script>"""
    document='<!doctype html><meta charset="utf-8"><title>Cartoonify 盲评</title><style>body{background:#171322;color:#eee;font:16px system-ui;margin:30px}img{width:100%}article{margin:30px 0}button{padding:8px 18px}</style><h1>新旧版本盲评</h1><p>顺序随机固定。评估脸部自然度、色块完整性、线条和风格表现；程序未替用户打分。投票仅保存在本机浏览器。</p><button onclick="download()">导出评分 JSON</button>'+''.join(blind_cards)+script
    (out/'blind_review.html').write_text(document,encoding='utf-8')
    reference=ROOT/'samples/high_resolution/reference_4k.jpg'
    if reference.exists():
        original=load_image(reference)
        for index,style in enumerate(STYLES):
            result=process(original,style);Image.fromarray(result).save(out/f'highres_{index+1}.png',compress_level=3)
            # Native-detail crop, not a thumbnail: compare with v1 operating on this real 4K input.
            old=legacy.process(original,style)
            h,w=original.shape[:2];y=max(0,h//2-260);x=max(0,w//2-260)
            sheet([original[y:y+520,x:x+520],old[y:y+520,x:x+520],result[y:y+520,x:x+520]],['Original native crop','Before v1','After v2'],comparisons/f'real4k_{index+1}.jpg',height=560)
    build_gallery()
    print('Generated 84 outputs, 21 sheets, 24 A/B panels and 4 real-4K native crops',flush=True)

if __name__=='__main__': main()
