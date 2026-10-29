"""Full CPU anime evaluation; compare against immutable published artifacts."""
from pathlib import Path
from dataclasses import replace
from io import BytesIO
import hashlib,html,json,subprocess,time,threading
import numpy as np,psutil
from PIL import Image
from cartoonify.core import process,load_image,resize,Params,PRESETS,STYLES,CACHE
from evaluate import sheet
from build_gallery import main as build_gallery
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/anime_v7'
BASELINE='f58311c6696f285c62be335c00830fcbb190c668'


def main():
    OUT.mkdir(exist_ok=True);rows=[];cards=[]
    unchanged={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for i in (1,2,3) for p in (ROOT/f'outputs/{i+1}_{STYLES[i]}').glob('*.png')}
    samples=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    selected={'01_portrait','02_portrait','03_portrait','05_portrait','13_portrait','23_portrait'}
    proc=psutil.Process()
    for record in samples:
        path=ROOT/'samples'/record['filename'];stem=path.stem;image=load_image(path)
        old=np.array(Image.open(BytesIO(subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASELINE}:outputs/1_经典卡通/{stem}.png']))).convert('RGB'))
        CACHE.clear();info={};peak=[proc.memory_info().rss];stop=threading.Event()
        def monitor():
            while not stop.wait(.01):peak[0]=max(peak[0],proc.memory_info().rss)
        thread=threading.Thread(target=monitor,daemon=True);thread.start();t=time.perf_counter()
        try:result=process(image,STYLES[0],diagnostics=info)
        finally:stop.set();thread.join()
        seconds=time.perf_counter()-t
        Image.fromarray(result).save(OUT/f'{stem}.png',compress_level=3)
        Image.fromarray(result).save(ROOT/f'outputs/1_经典卡通/{stem}.png',compress_level=3)
        t=time.perf_counter();preview=process(image,STYLES[0],preview=True,quality='标准');cached=time.perf_counter()-t
        difference=float(np.abs(preview.astype(float)-resize(result,1000).astype(float)).mean())
        assert difference<4,(stem,difference)
        sheet([image,old,result],['Original','Published FacePaint v2','Anime-enhanced Webtoon'],OUT/f'{stem}_review.jpg',height=600)
        panels=[image,result]+[load_image(ROOT/f'outputs/{i+1}_{STYLES[i]}/{stem}.png') for i in (1,2,3)]
        sheet(panels,['Original','Anime v7','Flat','Comic','Pencil v6'],ROOT/f'outputs/{stem}_comparison.jpg',height=420)
        if info['faces']:
            x,y,bw,bh=max(info['faces'],key=lambda b:b[2]*b[3]);scale=max(image.shape[:2])/1000 if max(image.shape[:2])>1000 else 1
            left=max(0,round((x-bw*.25)*scale));top=max(0,round((y-bh*.25)*scale));right=min(image.shape[1],round((x+bw*1.25)*scale));bottom=min(image.shape[0],round((y+bh*1.25)*scale))
            if right>left and bottom>top:sheet([a[top:bottom,left:right] for a in (image,old,result)],['Original','Published v2','Anime v7'],OUT/f'{stem}_crop.png',height=480)
        row={'file':path.name,'seconds':seconds,'cached_preview_seconds':cached,'peak_process_rss_mib':peak[0]/1024**2,'preview_export_mae':difference,'shape':list(result.shape),'anime':info['anime'],'stages':info['stages'],'sha256':hashlib.sha256((OUT/f'{stem}.png').read_bytes()).hexdigest()}
        rows.append(row)
        cards.append(f'<article><h2>{html.escape(path.name)}</h2><img loading="lazy" src="{stem}_review.jpg"><p>{seconds:.2f} 秒 · 局部人脸 {info["anime_refined_faces"]} · <a href="{stem}.png">原尺寸 PNG</a></p><details><summary>来源署名</summary>{html.escape(record["author"])} · {html.escape(record["license"])} · <a href="{html.escape(record["source_url"],quote=True)}">来源</a></details></article>')
        (OUT/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'sample':path.name,'seconds':round(seconds,2),'faces':info['anime_refined_faces'],'mae':difference}),flush=True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in unchanged.items())
    (OUT/'unchanged_other_styles.json').write_text(json.dumps(unchanged,ensure_ascii=False,indent=2),encoding='utf-8')
    doc='<!doctype html><meta charset="utf-8"><title>动漫人像 v7</title><style>body{background:#171322;color:#eee;font:16px system-ui;max-width:1500px;margin:30px auto;padding:20px}img{width:100%}article{margin:30px 0}a{color:#c8adff}</style><h1>动漫人像 v7：原图 / 已发布柔和插画 / 动漫增强</h1><p>21 张全量样片。模型可能改变身份、年龄感和肤色；照片及署名见各卡片。程序性能测试不代表审美评分。</p><p><a href="../../anime_v7_report.md">CPU 性能与已知局限</a> · <a href="../gallery.html">全部四种风格</a></p>'+''.join(cards)
    (OUT/'gallery.html').write_text(doc,encoding='utf-8');build_gallery()


if __name__=='__main__':main()
