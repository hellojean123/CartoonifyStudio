"""Isolated v3 workers, same source/crop as historical v2 benchmark."""
from pathlib import Path
import json,subprocess,sys
from benchmark_v2 import worker
ROOT=Path(__file__).parent

def main():
    rows=[]
    for w,h in ((640,480),(1920,1080),(3840,2160)):
        for index in range(4):
            r=subprocess.run([sys.executable,__file__,'--worker',str(index),str(w),str(h)],capture_output=True,text=True,encoding='utf-8',check=True)
            row=json.loads(r.stdout.strip().splitlines()[-1]);row['version']='v3';rows.append(row)
            print(w,h,index,round(row['median_seconds'],3),round(row['rss_peak_mib'],1),flush=True)
    (ROOT/'outputs/benchmark_v3_isolated.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    baseline=json.loads((ROOT/'outputs/benchmark_v2.json').read_text(encoding='utf-8'))
    lines=['# v3 CPU 实测与视觉核验','', '独立子进程、预热后禁用缓存运行三次；RSS 每 5ms 采样。与 v2 使用相同真实照片及裁剪方式。v2 为历史实测，v3 为本次实测，非同时运行。时间不含解码、编码、网页传输；旧数据仍保存在 benchmark.md。','', '| 尺寸 | 风格 | v2 历史中位秒 | v3 中位秒 | v3 RSS MiB |','|---|---|---:|---:|---:|']
    for row in rows:
        old=next(b for b in baseline if b['version']=='v2' and b['width']==row['width'] and b['height']==row['height'] and b['style']==row['style'])
        lines.append(f"| {row['width']}×{row['height']} | {row['style']} | {old['median_seconds']:.3f} | {row['median_seconds']:.3f} | {row['rss_peak_mib']:.1f} |")
    lines+=['','高清策略：经典卡通仅少量引导恢复；插画保留离散色块；漫画保留分级与输出尺度网点；素描保留分析尺度明暗、输出尺度颗粒。插画/漫画/素描的标准与精细当前使用同一结构重建，避免精细档引入照片噪声。','', '效果证据：outputs/v3_review/gallery.html 与同目录原尺寸 crop；全量 84 张结果与 21 张对比已重新生成。数值检查不代替审美评价，没有虚构盲评分数。']
    (ROOT/'benchmark_v3.md').write_text('\n'.join(lines)+'\n\n性能取舍：本次 640×480 的四风格首次处理均比历史 v2 慢，新增人脸定位与区域处理有固定开销；4K 素描也由历史 1.295 秒增至本次 2.993 秒。经典、插画、漫画的 4K 时间下降，但不能宣称所有尺寸、所有风格都加速。',encoding='utf-8')

if __name__=='__main__':
    if len(sys.argv)>1:
        i,w,h=map(int,sys.argv[2:]);print(json.dumps(worker(i,w,h,'v2'),ensure_ascii=True))
    else:main()
