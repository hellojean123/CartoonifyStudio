"""Repeatable isolated-worker CPU benchmark using genuine 4K source detail."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
import psutil
from cartoonify.core import STYLES,process,load_image,CACHE,Params,PRESETS
from cartoonify.service import encode

ROOT=Path(__file__).parent

def worker(style_index,width,height,version):
    start=time.perf_counter();source=load_image(ROOT/'samples/high_resolution/reference_4k.jpg');decode=time.perf_counter()-start
    # Resize a genuinely detailed source, preserving aspect by center cropping.
    scale=max(width/source.shape[1],height/source.shape[0]);scaled=cv2.resize(source,(round(source.shape[1]*scale),round(source.shape[0]*scale)),interpolation=cv2.INTER_AREA)
    y=(scaled.shape[0]-height)//2;x=(scaled.shape[1]-width)//2;image=scaled[y:y+height,x:x+width].copy()
    del source,scaled
    style=STYLES[style_index]
    if version=='v1':
        from benchmarks import legacy_v1 as legacy
        fn=lambda:legacy.process(image,style)
    else: fn=lambda:process(image,style,cache=None)
    # Warm native libraries; cold here means intermediate cache disabled, not fresh DLLs.
    temp=fn();del temp
    proc=psutil.Process();rss_base=proc.memory_info().rss;peaks=[rss_base];stop=threading.Event()
    def monitor():
        while not stop.wait(.005): peaks.append(proc.memory_info().rss)
    thread=threading.Thread(target=monitor);thread.start();times=[]
    try:
        for _ in range(3):
            started=time.perf_counter();output=fn();times.append(time.perf_counter()-started)
            del output
    finally: stop.set();thread.join()
    warm=[];warm_preview=[];stage={}
    if version=='v2':
        CACHE.clear();output=process(image,style,diagnostics=stage)
        p=Params(*PRESETS[style_index])
        from dataclasses import replace
        for i in range(3):
            started=time.perf_counter();output=process(image,style,replace(p,outline=p.outline-.05*(i+1)));warm.append(time.perf_counter()-started)
        for i in range(3):
            started=time.perf_counter();preview=process(image,style,replace(p,outline=p.outline-.05*(i+1)),preview=True,quality='标准');warm_preview.append(time.perf_counter()-started)
        started=time.perf_counter();data=encode(output,'PNG');encoding=time.perf_counter()-started
        encoded_size=len(data)
    else: encoding=None;encoded_size=None
    return {'version':version,'style':style,'width':width,'height':height,'runs':times,'median_seconds':float(np.median(times)),'p95_seconds':float(np.percentile(times,95)),'rss_peak_mib':max(peaks)/1024**2,'rss_increase_mib':(max(peaks)-rss_base)/1024**2,'cached_parameter_runs':warm,'cached_parameter_median':float(np.median(warm)) if warm else None,'cached_preview_runs':warm_preview,'cached_preview_median':float(np.median(warm_preview)) if warm_preview else None,'png_encoding_seconds':encoding,'encoded_bytes':encoded_size,'decode_reference_seconds':decode,'stages':stage.get('stages',{})}

def main():
    rows=[]
    for width,height in [(640,480),(1280,720),(1920,1080),(3840,2160)]:
        for style_index in range(4):
            for version in ('v1','v2'):
                command=[sys.executable,__file__,'--worker','--style',str(style_index),'--width',str(width),'--height',str(height),'--version',version]
                result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
                row=json.loads(result.stdout.strip().splitlines()[-1]);rows.append(row)
                print(width,height,style_index,version,round(row['median_seconds'],3),flush=True)
    (ROOT/'outputs/benchmark_v2.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    from cartoonify.core import quantize,resize
    from benchmarks import legacy_v1 as legacy
    image=resize(load_image(ROOT/'samples/high_resolution/reference_4k.jpg'),1000);quant=[]
    for name,fn in [('v1',legacy.quantize),('v2',quantize)]:
        temp=fn(image,16);del temp;runs=[]
        for _ in range(3):
            started=time.perf_counter();temp=fn(image,16);runs.append(time.perf_counter()-started);del temp
        quant.append({'version':name,'runs':runs,'median_seconds':float(np.median(runs))})
    (ROOT/'outputs/quantization_benchmark.json').write_text(json.dumps(quant,indent=2),encoding='utf-8')
    lines=['# CPU 性能对照实测（v2）','',f'Python {platform.python_version()}，OpenCV {cv2.__version__}，{platform.platform()}，{platform.processor()}，逻辑 CPU {psutil.cpu_count()}，OpenCV 线程 2，OpenCL 关闭。',
           '', '来源：samples/high_resolution/reference_4k.jpg，真实 3840×2880 相机照片缩放并中心裁剪；未将低清照片放大冒充高清。v1/v2 使用相同输入。',
           '', '每个分辨率、算法、版本使用独立子进程；预热后禁用中间缓存运行 3 次。P95 为这 3 次的描述性百分位，样本少，不表示长期尾延迟。RSS 为 5ms 采样的绝对进程峰值，含解释器/输入/分配器缓存；另存增量，采样可能漏掉瞬时尖峰。计时不含解码和编码；JSON 另记真实源图解码、PNG 编码和各阶段时间。',
           '', '| 分辨率 | 风格 | v1 中位秒 | v2 中位秒 | v2 P95 秒 | v2 调轮廓高清缓存秒 | v2 最终预览缓存秒 | v2 RSS MiB |','|---|---|---:|---:|---:|---:|---:|---:|']
    for i in range(0,len(rows),2):
        old,new=rows[i:i+2]
        lines.append(f'| {new["width"]}×{new["height"]} | {new["style"]} | {old["median_seconds"]:.3f} | {new["median_seconds"]:.3f} | {new["p95_seconds"]:.3f} | {new["cached_parameter_median"]:.3f} | {new["cached_preview_median"]:.3f} | {new["rss_peak_mib"]:.1f} |')
    lines+=['',f'独立量化实测（相同 1000 边长输入、16 色、预热后 3 次中位数）：v1 {quant[0]["median_seconds"]:.3f} 秒，v2 {quant[1]["median_seconds"]:.3f} 秒。v2 包含 Lab 采样聚类、32³ LUT 与空间投票；不是逐像素等价替换。',
            '', 'v1 为最长边 1400 处理后插值，v2 含人像检测、统一 1000 分析图、原图引导重建、输出尺度细线与网点。功能不完全相同，因此速度差不代表单一优化的因果收益。缓存值使用完整输入且只改变轮廓强度；不包括网页传输。',
            '', '内存策略：112 MiB 中间缓存 + 64 MiB 解码缓存分别封顶；高清重建按 512 图块带重叠边界处理，仍保留输入和最终 RGB 输出。资源共享队列与应用锁保证重任务串行。1600 万像素 / 40 MB 上限保留。',
            '', '旧版算法参考位于 benchmarks/legacy_v1.py；新旧实测数据见 outputs/benchmark_v2.json；全量照片与预览一致性见 outputs/timings.json。']
    (ROOT/'benchmark.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');parser.add_argument('--style',type=int);parser.add_argument('--width',type=int);parser.add_argument('--height',type=int);parser.add_argument('--version');args=parser.parse_args()
    if args.worker: print(json.dumps(worker(args.style,args.width,args.height,args.version),ensure_ascii=True))
    else: main()
