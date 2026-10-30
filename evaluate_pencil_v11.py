from pathlib import Path
import sys,time,json,html,threading,os
import psutil,numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from cartoonify.core import process,load_image,Params,PRESETS,STYLES,resize,CACHE
out=root/'outputs/pencil_v11';records=json.loads((root/'samples/metadata.json').read_text(encoding='utf-8'))['images']
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);rows=[];cards=[]
for r in records:
    stem=Path(r['filename']).stem;a=load_image(root/'samples'/r['filename']);diag={};CACHE.clear();baseline=psutil.Process().memory_info().rss;peak=[baseline];stop=threading.Event()
    def monitor():
        while not stop.wait(.02):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();start=time.perf_counter()
    try:b=process(a,STYLES[3],Params(*PRESETS[3]),diagnostics=diag)
    finally:stop.set();watcher.join()
    elapsed=time.perf_counter()-start;assert a.shape==b.shape and b.dtype==np.uint8
    Image.fromarray(b).save(out/f'{stem}.png');Image.fromarray(b).save(root/'outputs/4_铅笔素描'/f'{stem}.png')
    old=Image.open(out/'before'/f'{stem}.png');canvas=Image.new('RGB',(1350,640),'white');d=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate([(Image.fromarray(a),'原图'),(old,'旧版'),(Image.fromarray(b),'方向流逐笔')]):
        im.thumbnail((445,600));canvas.paste(im,(i*450+(450-im.width)//2,38));d.text((i*450+8,4),label,font=font,fill='black')
    canvas.save(out/f'{stem}_compare.jpg',quality=93)
    # Refresh all-style comparisons without recomputing unaffected algorithms.
    combined=Image.new('RGB',(1600,390),'#eeeeee');d=ImageDraw.Draw(combined)
    for i,label in enumerate(['原图','动漫人像','扁平插画','夸张手绘','方向流素描']):
        im=Image.fromarray(a) if i==0 else Image.open(root/'outputs'/f'{i}_{STYLES[i-1]}'/f'{stem}.png')
        im.thumbnail((316,350));combined.paste(im,(i*320+(320-im.width)//2,38));d.text((i*320+8,4),label,font=font,fill='black')
    combined.save(root/'outputs'/f'{stem}_comparison.jpg',quality=92)
    row={'file':r['filename'],'person':r.get('person',''),'size':list(a.shape[1::-1]),'seconds':round(elapsed,3),'baseline_rss_mib':round(baseline/2**20,1),'peak_rss_mib':round(peak[0]/2**20,1),'diagnostics':diag};rows.append(row)
    (out/'evaluation.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'file':r['filename'],'seconds':row['seconds'],'stages':diag['stages'],'strokes':diag['pencil_brush']['stroke_count']}),flush=True)
    cards.append(f'<article><h2>{html.escape(r.get("person",r["filename"]))}</h2><a href="{stem}_compare.jpg"><img loading="lazy" src="{stem}_compare.jpg"></a><p>{html.escape(r["author"])} · <a href="{html.escape(r["source_url"],quote=True)}">来源</a> · {html.escape(r["license"])}</p></article>')
document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>方向流逐笔素描对照</title><style>body{background:#1a1722;color:#eee;font:16px system-ui;max-width:1450px;margin:auto;padding:24px}img{width:100%}article{margin:24px 0;padding:16px;background:#292433}a{color:#d3bdff}</style><h1>AAAI 2021 思路 CPU 适配 · 33 张新旧对照</h1><p>左：原图；中：旧版；右：ETF方向流、灰度分层、有序笔画。非论文原样复现；派生图片保留来源许可。</p>'+''.join(cards)+'</html>'
(out/'gallery.html').write_text(document,encoding='utf-8')
import build_gallery;build_gallery.main()
print('COMPLETE',len(rows),flush=True)
