"""Reproducible CPU pencil evaluation against the exact published v4 artifacts."""
from pathlib import Path
from io import BytesIO
from dataclasses import replace
import argparse,hashlib,html,json,subprocess,sys,time,threading
import cv2,numpy as np,psutil
from PIL import Image,ImageOps,ImageDraw
from cartoonify.core import process,load_image,resize,STYLES,Params,PRESETS,CACHE
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pencil_v5'
BASELINE='0a2cb624facc1dd3bca249c1ec9df421d8c71544'


def sheet(panels,labels,dest,width=400):
    height=max(round(a.shape[0]*width/a.shape[1]) for a in panels)
    canvas=Image.new('RGB',(len(panels)*width,height+32),'#eeeeee');draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate(zip(panels,labels)):
        tile=ImageOps.contain(Image.fromarray(a),(width,height))
        canvas.paste(tile,(i*width+(width-tile.width)//2,32));draw.text((i*width+8,9),label,fill='black')
    canvas.save(dest)


def benchmark_one(size):
    source=load_image(ROOT/'samples/23_portrait.jpg')
    image=cv2.resize(source,(round(size*source.shape[1]/source.shape[0]),size),interpolation=cv2.INTER_AREA)
    proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    runs=[];stages=[]
    try:
        for _ in range(3):
            info={};started=time.perf_counter();process(image,STYLES[3],cache=None,diagnostics=info)
            runs.append(time.perf_counter()-started);stages.append(info['stages'])
        CACHE.clear();process(image,STYLES[3])
        started=time.perf_counter();process(image,STYLES[3],replace(Params(*PRESETS[3]),pencil_shade=.6),preview=True)
        cached=time.perf_counter()-started
    finally:stop.set();thread.join()
    return {'shape':list(image.shape),'uncached_seconds':runs,'median_seconds':float(np.median(runs)),
            'stages':stages,'cached_parameter_preview_seconds':cached,'peak_process_rss_mib':peak[0]/1024**2,
            'rss_sampling_ms':10,'opencv_threads':cv2.getNumThreads(),'opencl':cv2.ocl.useOpenCL()}


def main():
    OUT.mkdir(exist_ok=True);cards=[];records=[]
    other_files=[p for i in range(3) for p in (ROOT/f'outputs/{i+1}_{STYLES[i]}').glob('*.png')]
    before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in other_files}
    samples=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    for record in samples:
        path=ROOT/'samples'/record['filename'];im=load_image(path);stem=path.stem
        info={};CACHE.clear();started=time.perf_counter();result=process(im,STYLES[3],diagnostics=info)
        seconds=time.perf_counter()-started
        old_data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{BASELINE}:outputs/4_铅笔素描/{stem}.png'])
        old=np.array(Image.open(BytesIO(old_data)).convert('RGB'))
        Image.fromarray(result).save(OUT/f'{stem}.png',compress_level=3)
        small=resize(im,1000);t=time.perf_counter()
        baseline,_=cv2.pencilSketch(cv2.cvtColor(small,cv2.COLOR_RGB2BGR),sigma_s=60,sigma_r=.07,shade_factor=.05)
        baseline_seconds=time.perf_counter()-t
        Image.fromarray(baseline).save(OUT/f'{stem}_opencv.png')
        review=f'{stem}_review.jpg'
        sheet([resize(a,720) for a in (im,old,result)]+[cv2.cvtColor(baseline,cv2.COLOR_GRAY2RGB)],
              ['Original','Published v4','Pencil v5','OpenCV 60/.07/.05'],OUT/review)
        preview=process(im,STYLES[3],preview=True)
        error=float(np.abs(preview.astype(float)-resize(result,1000).astype(float)).mean())
        row={'file':path.name,'shape':list(im.shape),'seconds':seconds,'stages':info['stages'],'faces':info['faces'],
             'preview_export_mae':error,'opencv_seconds_1000':baseline_seconds,
             'sha256':hashlib.sha256((OUT/f'{stem}.png').read_bytes()).hexdigest()}
        detail=''
        if info['faces']:
            x,y,bw,bh=max(info['faces'],key=lambda b:b[2]*b[3]);ah,aw=info['analysis_size']
            x,y,bw,bh=[round(v*r) for v,r in zip((x,y,bw,bh),(im.shape[1]/aw,im.shape[0]/ah,im.shape[1]/aw,im.shape[0]/ah))]
            pad=round(bw*.1);x=max(0,x-pad);y=max(0,y-pad);bw=min(bw+pad*2,im.shape[1]-x);bh=min(bh+pad*2,im.shape[0]-y)
        else:
            bh=min(400,im.shape[0]);bw=min(400,im.shape[1]);x=(im.shape[1]-bw)//2;y=(im.shape[0]-bh)//2
        crop=f'{stem}_crop.png';row['native_crop_xywh']=[x,y,bw,bh]
        sheet([a[y:y+bh,x:x+bw] for a in (im,old,result)],['Original detail','v4 detail','v5 detail'],OUT/crop,width=bw)
        detail=f'<details><summary>原始像素局部</summary><a href="{crop}"><img loading="lazy" src="{crop}"></a></details>'
        colored=process(im,STYLES[3],replace(Params(*PRESETS[3]),colored=True))
        Image.fromarray(colored).save(OUT/f'{stem}_colored.png',compress_level=3)
        cards.append(f'<article><h2>{html.escape(stem)}</h2><img loading="lazy" src="{review}">{detail}<p>{seconds:.3f} 秒 · 预览/导出 MAE {error:.3f}/255 · <a href="{stem}.png">原尺寸素描</a> · <a href="{stem}_colored.png">原尺寸彩铅</a></p></article>')
        Image.fromarray(result).save(ROOT/f'outputs/4_铅笔素描/{stem}.png',compress_level=3)
        panels=[im]+[np.array(Image.open(ROOT/f'outputs/{i+1}_{STYLES[i]}/{stem}.png').convert('RGB')) for i in range(3)]+[result]
        sheet([resize(a,600) for a in panels],['Original','Anime','Flat','Comic','Pencil v5'],ROOT/f'outputs/{stem}_comparison.jpg',360)
        records.append(row);print(stem,round(seconds,3),flush=True)
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in other_files}
    assert before==after,'Another style output changed'
    (OUT/'unchanged_other_styles.json').write_text(json.dumps({'count':len(before),'sha256':before,'unchanged':before==after},indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'evaluation.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>铅笔素描 v5：明暗与笔触</title><style>body{max-width:1700px;margin:auto;padding:24px;font:16px system-ui;background:#17151e;color:#eee}article{background:#26232f;padding:18px;margin:24px 0;border-radius:12px}img{max-width:100%;height:auto}a{color:#c5b8ff}summary{cursor:pointer;padding:14px}p{line-height:1.8}</style><h1>铅笔素描：结构线 / 明暗 / 有限笔触</h1><p>每张均对比原图、准确的已发布 v4 结果、新版与 OpenCV pencilSketch 固定参数基线。基线不代表最优调参效果。局部按原照片像素裁剪；全部照片及修改结果沿用样片授权。</p><p><a href="../../samples/ATTRIBUTION.md">样片署名</a> · <a href="../../pencil_v5_report.md">算法、实测与局限</a> · <a href="../gallery.html">四风格总画廊</a></p>'''
    (OUT/'gallery.html').write_text(page+''.join(cards)+'</html>',encoding='utf-8')
    high=load_image(ROOT/'samples/high_resolution/reference_4k.jpg')
    Image.fromarray(process(high,STYLES[3])).save(ROOT/'outputs/highres_4.png',compress_level=3)
    (OUT/'highres_dimensions.json').write_text(json.dumps({'shape':list(high.shape),'output':'../highres_4.png'}),encoding='utf-8')
    benchmarks=[]
    for size in (640,1080,3840):
        data=subprocess.check_output([sys.executable,__file__,'--benchmark-one',str(size)],cwd=ROOT)
        benchmarks.append(json.loads(data))
    (OUT/'benchmark.json').write_text(json.dumps(benchmarks,indent=2),encoding='utf-8')
    from build_gallery import main as build_gallery
    build_gallery();gallery=ROOT/'outputs/gallery.html';s=gallery.read_text(encoding='utf-8')
    s=s.replace('<body>','<body><p><a href="pencil_v5/gallery.html">铅笔素描 v5：21 张新旧 / OpenCV 对照与原像素局部</a></p>')
    gallery.write_text(s,encoding='utf-8')
    print('21 full-size pencil + 21 colored, 21 reviews + native crops, 4K, 3 isolated CPU benchmarks complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-one',type=int);args=parser.parse_args()
    if args.benchmark_one:print(json.dumps(benchmark_one(args.benchmark_one)))
    else:main()
