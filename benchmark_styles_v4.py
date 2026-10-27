"""CPU wall-clock and sampled absolute RSS, including first neural session load."""
from pathlib import Path
import sys,time,json,subprocess,threading,platform
import numpy as np
import cv2,psutil
from cartoonify.core import process,load_image,resize,STYLES,CACHE
ROOT=Path(__file__).resolve().parent

def worker(index,w,h):
    source=load_image(ROOT/'samples/high_resolution/reference_4k.jpg')
    ratio=max(w/source.shape[1],h/source.shape[0]);source=cv2.resize(source,(round(source.shape[1]*ratio),round(source.shape[0]*ratio)),interpolation=cv2.INTER_AREA)
    y=(source.shape[0]-h)//2;x=(source.shape[1]-w)//2;im=source[y:y+h,x:x+w].copy();del source
    proc=psutil.Process();initial=proc.memory_info().rss;peaks=[initial];stop=threading.Event()
    def monitor():
        while not stop.wait(.005):peaks.append(proc.memory_info().rss)
    monitor_thread=threading.Thread(target=monitor);monitor_thread.start();runs=[]
    try:
        started=time.perf_counter();output=process(im,STYLES[index],cache=None);cold=time.perf_counter()-started;del output
        for _ in range(3):
            started=time.perf_counter();output=process(im,STYLES[index],cache=None);runs.append(time.perf_counter()-started);del output
    finally:stop.set();monitor_thread.join()
    CACHE.clear();process(im,STYLES[index]);started=time.perf_counter();process(im,STYLES[index],preview=True);cached=time.perf_counter()-started
    return {'style':STYLES[index],'width':w,'height':h,'cold_seconds':cold,'warm_no_image_cache_runs':runs,'median_seconds':float(np.median(runs)),
        'cached_preview_seconds':cached,'rss_peak_mib':max(peaks)/1024**2,'rss_increase_mib':(max(peaks)-initial)/1024**2}

def main():
    rows=[]
    for w,h in ((640,480),(1920,1080),(3840,2160)):
        for i in range(4):
            r=subprocess.run([sys.executable,__file__,'--worker',str(i),str(w),str(h)],capture_output=True,text=True,encoding='utf-8',check=True)
            row=json.loads(r.stdout.strip().splitlines()[-1]);rows.append(row);print(w,h,i,round(row['median_seconds'],3),round(row['rss_peak_mib'],1),flush=True)
    (ROOT/'outputs/benchmark_v4.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'opencv':cv2.__version__,'threads':2,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 四风格重做 CPU 实测（2026-10-27）','','每项在独立进程运行，真实 4K 山景缩放并中心裁剪。冷启动包含该进程首次模型加载；预热后禁用图像中间缓存连续 3 次，报告中位数。RSS 从首次算法调用前起每 5ms 采样，含 ONNX 会话、输入和分配器；采样可能漏瞬时尖峰。计时不含文件编解码、网络与 UI。只有动漫使用预训练绘画模型，所有推理强制 CPUExecutionProvider；OpenCV 2 线程、OpenCL 关闭。','','| 尺寸 | 风格 | 冷启动秒 | 预热中位秒 | 缓存预览秒 | 进程峰值 MiB |','|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['width']}×{r['height']} | {r['style']} | {r['cold_seconds']:.3f} | {r['median_seconds']:.3f} | {r['cached_preview_seconds']:.3f} | {r['rss_peak_mib']:.1f} |")
    lines+=['','动漫最大 640 边长推理后重建到统一 1000 分析图，再按原尺寸分块导出；不会用原照片锐化冒充新增动漫细节。素描的结构/笔触只算一次，输出使用统一坐标，避免图块接缝。中间缓存 112 MiB、解码缓存 64 MiB；模型内存单独计算，未把这两个缓存上限说成进程总内存上限。','', '增加动漫画风需要付出比旧量化滤镜更高的 CPU 耗时；先看质量，再看性能。本报告没有把算法变化包装为等质量加速，也没有虚构审美分数。原始各次时间见 outputs/benchmark_v4.json。']
    (ROOT/'benchmark_v4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    if len(sys.argv)>1:print(json.dumps(worker(*map(int,sys.argv[2:]))))
    else:main()
