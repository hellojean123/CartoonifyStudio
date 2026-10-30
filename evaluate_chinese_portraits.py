from pathlib import Path
import sys,time,json,html,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from cartoonify.core import process,load_image,STYLES,PRESETS,Params
out=root/'outputs/chinese_portraits_v10';out.mkdir(exist_ok=True)
records=json.loads((root/'samples/metadata.json').read_text(encoding='utf-8'))['images'];rows=[];cards=[]
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
for r in records:
    if r.get('collection')!='chinese_celebrities':continue
    a=load_image(root/'samples'/r['filename']);stem=Path(r['filename']).stem;images=[a];stats=[]
    for i,style in enumerate(STYLES):
        diag={};t=time.perf_counter();b=process(a,style,Params(*PRESETS[i]),diagnostics=diag);elapsed=time.perf_counter()-t
        assert b.shape==a.shape and b.dtype==np.uint8 and np.isfinite(b).all()
        dest=root/'outputs'/f'{i+1}_{style}'/f'{stem}.png';Image.fromarray(b).save(dest);images.append(b)
        stats.append({'style':style,'seconds':round(elapsed,3),'faces':len(diag.get('faces',[])),'warped_faces':diag.get('caricature',{}).get('warped_faces',0),'output':str(dest.relative_to(root))})
    canvas=Image.new('RGB',(1600,390),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate(zip(images,['原图','动漫人像','扁平插画','夸张手绘','铅笔素描'])):
        im=Image.fromarray(a);im.thumbnail((316,350));canvas.paste(im,(i*320+(320-im.width)//2,38));draw.text((i*320+8,6),label,font=font,fill='#222222')
    canvas.save(root/'outputs'/f'{stem}_comparison.jpg',quality=92)
    rows.append({'filename':r['filename'],'person':r['person'],'dimensions':list(images[0].shape[:2][::-1]),'results':stats})
    cards.append(f'<article><h2>{html.escape(r["person"])}</h2><a href="../{stem}_comparison.jpg"><img src="../{stem}_comparison.jpg" loading="lazy"></a><p>{html.escape(r["author"])} · <a href="{html.escape(r["license_url"],quote=True)}">{html.escape(r["license"])}</a> · <a href="{html.escape(r["source_url"],quote=True)}">原始来源</a></p></article>')
    (out/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows[-1]),flush=True)
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>华语明星四风格实测</title><style>body{background:#191622;color:#eee;font:16px system-ui;max-width:1680px;margin:auto;padding:24px}img{width:100%}article{margin:24px 0;padding:16px;background:#282333;border-radius:12px}a{color:#cdb5ff}</style><h1>新增华语明星 · 四风格效果检查</h1><p>所有效果均由本项目本地 CPU 生成；派生图片保留来源许可证。</p>'+''.join(cards)+'</html>'
(out/'gallery.html').write_text(doc,encoding='utf-8')
import build_gallery;build_gallery.main()
print('COMPLETE',len(rows),flush=True)
