from pathlib import Path
from dataclasses import replace
from io import BytesIO
import hashlib,html,json,subprocess,time
import numpy as np
from PIL import Image
from cartoonify.core import process,load_image,resize,Params,PRESETS,STYLES,CACHE
from evaluate import sheet
from build_gallery import main as build_gallery
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/caricature_v9';BASE='fea48c09ff47cc4e2c61e5d690f6623f72fa9b16'


def main():
    OUT.mkdir(exist_ok=True);rows=[];cards=[];selected={'01_portrait','03_portrait','05_portrait','23_portrait','26_portrait','29_portrait'}
    unchanged={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for i in (0,1,3) for p in (ROOT/f'outputs/{i+1}_{STYLES[i]}').glob('*.png')}
    samples=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    for record in samples:
        path=ROOT/'samples'/record['filename'];stem=path.stem;im=load_image(path);CACHE.clear();p=Params(*PRESETS[2]);info={}
        old=np.array(Image.open(BytesIO(subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASE}:outputs/3_美式漫画/{stem}.png']))).convert('RGB'))
        zero=process(im,STYLES[2],replace(p,caricature_strength=0))
        Image.fromarray(zero).save(OUT/f'{stem}_zero.png',compress_level=3)
        t=time.perf_counter();result=process(im,STYLES[2],p,diagnostics=info);seconds=time.perf_counter()-t
        for dest in (OUT/f'{stem}.png',ROOT/f'outputs/3_美式漫画/{stem}.png'):Image.fromarray(result).save(dest,compress_level=3)
        t=time.perf_counter();preview=process(im,STYLES[2],p,preview=True,quality='标准');cached=time.perf_counter()-t
        mae=float(np.abs(preview.astype(float)-resize(result,1000).astype(float)).mean());assert mae<4,(stem,mae)
        sheet([im,old,zero,result],['Original','Published print comic','New ink, no enlargement','Big nose 0.75'],OUT/f'{stem}_review.jpg',width=1800,height=620)
        if info['faces']:
            x,y,bw,bh=max(info['faces'],key=lambda b:b[2]*b[3]);scale=max(im.shape[:2])/1000 if max(im.shape[:2])>1000 else 1
            x0=max(0,round((x-bw*.2)*scale));x1=min(im.shape[1],round((x+bw*1.2)*scale));y0=max(0,round((y-bh*.2)*scale));y1=min(im.shape[0],round((y+bh*1.2)*scale))
            if x1>x0 and y1>y0:sheet([a[y0:y1,x0:x1] for a in (im,old,zero,result)],['Photo','Published','Ink 0','Nose 0.75'],OUT/f'{stem}_crop.png',width=1600,height=480)
        modes=[]
        if stem in selected:
            modes=[result]
            for feature,key in [('眼睛','eyes'),('笑容','smile'),('额头','forehead')]:
                alt=process(im,STYLES[2],replace(p,caricature_feature=feature));modes.append(alt)
                Image.fromarray(alt).save(OUT/f'{stem}_{key}.png',compress_level=3)
            sheet([im,*modes],['Photo','Big nose','Big eyes','Wide smile','Big forehead'],OUT/f'{stem}_features.jpg',width=1800,height=600)
        panels=[im]+[result if i==2 else load_image(ROOT/f'outputs/{i+1}_{STYLES[i]}/{stem}.png') for i in range(4)]
        sheet(panels,['Original','Anime v7','Flat v8','Caricature v9','Pencil v6'],ROOT/f'outputs/{stem}_comparison.jpg',height=420)
        row={'file':path.name,'seconds':seconds,'cached_preview_seconds':cached,'preview_export_mae':mae,'caricature':info['caricature'],'zero_vs_exaggerated_mae':float(np.abs(zero.astype(float)-result.astype(float)).mean()),'sha256':hashlib.sha256((OUT/f'{stem}.png').read_bytes()).hexdigest()}
        rows.append(row);(OUT/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        extra=f'<p><a href="{stem}_features.jpg">四种特征放大对照</a></p>' if modes else ''
        cards.append(f'<article><h2>{html.escape(path.name)}</h2><img loading="lazy" src="{stem}_review.jpg"><p>实际变形 {info["caricature"]["warped_faces"]} 张脸 · {seconds:.2f} 秒 · <a href="{stem}.png">原尺寸 PNG</a></p>{extra}<details><summary>来源署名</summary>{html.escape(record["author"])} · {html.escape(record["license"])} · <a href="{html.escape(record["source_url"],quote=True)}">来源</a></details></article>')
        print(json.dumps({'sample':path.name,'seconds':round(seconds,2),'faces':info['caricature']['warped_faces'],'mae':mae}),flush=True)
    assert len(unchanged)==63 and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in unchanged.items())
    (OUT/'unchanged_other_styles.json').write_text(json.dumps(unchanged,ensure_ascii=False,indent=2),encoding='utf-8')
    doc='<!doctype html><meta charset="utf-8"><title>夸张手绘漫画 v9</title><style>body{background:#171322;color:#eee;font:16px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}article{margin:30px 0}a{color:#c8adff}</style><h1>原图 / 旧印刷漫画 / 新彩墨零变形 / 鼻子夸张 0.75</h1><p>没有定位到清晰人脸时只做彩墨绘制，不声称已扩大五官。自动最多处理三张脸；六张人像另有四种特征对照。</p><a href="../../caricature_v9_report.md">实际验证与局限</a>'+''.join(cards)
    (OUT/'gallery.html').write_text(doc,encoding='utf-8');build_gallery()


if __name__=='__main__':main()
