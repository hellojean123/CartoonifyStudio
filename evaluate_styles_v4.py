"""All licensed samples; compare to exact published pre-redesign commit."""
from pathlib import Path
import argparse,subprocess,time,json,html
from io import BytesIO
import numpy as np
from PIL import Image
from cartoonify.core import process,load_image,resize,STYLES,CACHE
from evaluate import sheet
ROOT=Path(__file__).resolve().parent
BASELINE='0b46ac10b7dd8174afe8502b769ee8f9bba409dd'
DISPLAY=['Anime portrait','Flat illustration','American comic','Pencil drawing']

def main(only_style=None):
    out=ROOT/'outputs';review=out/'v4_review';review.mkdir(exist_ok=True)
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    selected={'01_portrait','03_portrait','05_portrait','13_portrait','15_landscape','17_architecture','19_animal','23_portrait'}
    previous={}
    if only_style is not None:
        previous={(r['file'],r['style']):r for r in json.loads((out/'timings_v4.json').read_text(encoding='utf-8'))}
    rows=[];cards=[];comparisons=[]
    for record in records:
        path=ROOT/'samples'/record['filename'];im=load_image(path);panels=[im]
        for i,style in enumerate(STYLES):
            dest=out/f'{i+1}_{style}';dest.mkdir(exist_ok=True)
            fresh=only_style is None or only_style==i
            if fresh:
                CACHE.clear();info={};started=time.perf_counter();result=process(im,style,diagnostics=info);seconds=time.perf_counter()-started
                Image.fromarray(result).save(dest/(path.stem+'.png'),compress_level=3)
                preview=process(im,style,preview=True,quality='标准')
                rows.append({'file':path.name,'style':style,'display':DISPLAY[i],'seconds':seconds,'faces':info['faces'],'anime_refined_faces':info.get('anime_refined_faces',0),'stages':info['stages'],
                    'preview_export_mae':float(np.abs(preview.astype(float)-resize(result,1000)).mean()),
                    'paper_fraction':float(np.mean(np.all(result>245,axis=2))) if i==3 else None,
                    'dark_line_fraction':float(np.mean(np.all(result<90,axis=2))) if i==3 else None})
            else:
                with Image.open(dest/(path.stem+'.png')) as stored:result=np.array(stored.convert('RGB'))
                info=previous[(path.name,style)];rows.append(info)
                if result.shape!=im.shape:raise ValueError('Existing result shape mismatch')
            panels.append(result)
            if path.stem in selected:
                name=f'{path.stem}_{i+1}.jpg';crop=path.stem+f'_{i+1}_crop.png'
                if not fresh:
                    if not (review/name).exists() or not (review/crop).exists():raise ValueError('Missing previous comparison')
                    cards.append(f'<article data-style="{i}"><h2>{html.escape(path.stem)} · {DISPLAY[i]}</h2><a href="{name}"><img loading="lazy" src="{name}"></a><details><summary>放大五官 / 结构局部</summary><a href="{crop}"><img loading="lazy" src="{crop}"></a></details></article>')
                    continue
                # Compare to the exact published artifact, not a rerun under new dependencies.
                baseline_file=f'outputs/{i+1}_{style}/{path.stem}.png'
                data=subprocess.check_output(['git','show',BASELINE+':'+baseline_file],cwd=ROOT)
                with Image.open(BytesIO(data)) as baseline_image:before=np.array(baseline_image.convert('RGB'))
                if before.shape!=im.shape:raise ValueError('Published baseline shape mismatch: '+baseline_file)
                name=f'{path.stem}_{i+1}.jpg'
                sheet([im,before,result],['Original','Published old','Redesigned v4'],review/name,width=1500,height=650)
                h,w=im.shape[:2]
                if info['faces']:
                    box=max(info['faces'],key=lambda b:b[2]*b[3])
                    analysis_scale=min(1.,1000/max(h,w));aw=max(1,round(w*analysis_scale));ah=max(1,round(h*analysis_scale))
                    x,y,bw,bh=[round(v*scale) for v,scale in zip(box,(w/aw,h/ah,w/aw,h/ah))]
                    size=min(max(bw,bh),h,w,520)
                    x=int(np.clip(x+(bw-size)/2,0,w-size));y=int(np.clip(y+(bh-size)/2,0,h-size))
                else:size=min(420,h,w);x=(w-size)//2;y=(h-size)//2
                crop=path.stem+f'_{i+1}_crop.png'
                sheet([a[y:y+size,x:x+size] for a in (im,before,result)],['Original detail','Old detail','New detail'],review/crop,width=size*3+36,height=size+45)
                cards.append(f'<article data-style="{i}"><h2>{html.escape(path.stem)} · {DISPLAY[i]}</h2><a href="{name}"><img loading="lazy" src="{name}"></a><details><summary>放大五官 / 结构局部</summary><a href="{crop}"><img loading="lazy" src="{crop}"></a></details></article>')
        name=path.stem+'_comparison.jpg'
        sheet(panels,['Original',*DISPLAY],out/name,width=1800,height=500)
        comparisons.append(f'<article><h2>{html.escape(path.stem)}</h2><a href="../{name}"><img loading="lazy" src="../{name}"></a></article>')
        print('Rendered',path.name,flush=True)
    style_options=''.join(f'<option value="{i}">{name}</option>' for i,name in enumerate(['动漫人像','扁平插画','美式漫画','线条素描']))
    intro='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>四风格重做：真实照片核验</title><style>body{font:16px system-ui;margin:auto;padding:28px;max-width:1600px;background:#15131d;color:#eee}img{max-width:100%;height:auto}article{padding:20px;background:#24212e;border-radius:12px;margin:24px 0}a{color:#bcb1ff}select{padding:10px;font-size:16px;position:sticky;top:8px}summary{cursor:pointer;padding:12px}p{line-height:1.7}</style>'
    intro+='<h1>四种风格重做：原图 / 旧版 / 新版</h1><p>全部 21 张合规样片均重做四种风格。旧版固定为已发布提交 '+BASELINE[:10]+'。动漫使用本地 CPU FacePaint v2 预训练模型，会重绘人物特征；素描以留白、轮廓和顺结构笔触为主。高清尺寸输出不等于模型生成了原尺寸细节。没有自动审美评分。</p><p><a href="../../samples/ATTRIBUTION.md">照片署名与许可</a> · <a href="../../style_redesign_report.md">核验与局限</a> · <a href="../gallery.html">全部效果画廊</a></p>'
    script='<script>function filter(v){document.querySelectorAll("article[data-style]").forEach(e=>e.hidden=v!=="all"&&e.dataset.style!==v)}</script>'
    (review/'gallery.html').write_text(intro+'<h2>代表照片新旧对比（点击图片查看原尺寸）</h2><select onchange="filter(this.value)"><option value="all">全部风格</option>'+style_options+'</select>'+''.join(cards)+'<h2>全部 21 张：原图与四风格</h2>'+''.join(comparisons)+script+'</html>',encoding='utf-8')
    (out/'timings_v4.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    high=load_image(ROOT/'samples/high_resolution/reference_4k.jpg')
    for i,style in enumerate(STYLES):
        if only_style is not None and i!=only_style:continue
        CACHE.clear();Image.fromarray(process(high,style)).save(out/f'highres_{i+1}.png',compress_level=3)
    from build_gallery import main as build_gallery
    build_gallery()
    gallery=out/'gallery.html';text=gallery.read_text(encoding='utf-8')
    text=text.replace('<body>','<body><p><a href="v4_review/gallery.html">2026-10-27：动漫 / 线条素描等四风格重做，新旧与局部对比</a></p>')
    gallery.write_text(text,encoding='utf-8')
    print('84 full-size results, 21 comparisons, 32 old/new comparisons + face/detail crops; 4K exports done' if only_style is None else f'Regenerated style {only_style} for 21 samples; reused 63 unchanged results with their measured records; rebuilt comparisons and matching 4K export',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only-style',type=int,choices=range(4));args=parser.parse_args();main(args.only_style)
