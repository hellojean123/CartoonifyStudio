from pathlib import Path
import json
root=Path(__file__).resolve().parent;out=root/'outputs'
bench=root/'benchmark_v4.md';text=bench.read_text(encoding='utf-8')
marker='\n## 动漫模型内存池对照与人像细化成本\n'
if marker in text:text=text.split(marker)[0]
arena=json.loads((out/'anime_arena_benchmark.json').read_text(encoding='utf-8'))
off,on=arena
text+=marker+'\n另外以同一随机 1×3×480×640 张量、两个独立进程，各预热后 3 次比较 ONNX CPU 内存池。仅测推理，不含图像准备；与上表端到端时间口径不同。\n\n| 内存池 | 推理中位秒 | 进程峰值 MiB |\n|---|---:|---:|\n'
for row in arena:text+=f"| {'开启' if row['arena'] else '关闭'} | {row['median_seconds']:.3f} | {row['peak_mib']:.1f} |\n"
text+='\n两者耗时接近，开启后的峰值内存明显增加，因此最终继续关闭内存池。只有三次，不能将小幅时间差宣称为稳定加速。复现工具 benchmark_anime_arena.py；原始数据 outputs/anime_arena_benchmark.json。\n\n上方分辨率基准使用山景，没有覆盖新增人像局部重绘的全部成本。以下为最终逐图评估的实际人像用时：逐图清空图像缓存，复用 CPU 模型会话；每张仅一次，不冒充三次中位数，也不含编解码和网页传输。\n\n| 人像 | 实际局部细化脸数 | 本次处理秒 |\n|---|---:|---:|\n'
rows=json.loads((out/'timings_v4.json').read_text(encoding='utf-8'))
for name in ('02_portrait.jpg','03_portrait.jpg','23_portrait.jpg','26_portrait.jpg'):
    row=next(r for r in rows if r['file']==name and r['style']=='经典卡通')
    text+=f"| {name} | {row.get('anime_refined_faces',0)} | {row['seconds']:.3f} |\n"
text+='\n全图推理最长边 640，局部脸部上下文为 512，最多三张脸；输入分析图中不足 50 像素的脸不进行局部重绘。快速草稿跳过该步骤，最终预览和原尺寸导出使用同一份细化结构。\n'
bench.write_bytes(text.encode('utf-8'))
print('Added actual controlled memory tradeoff and portrait refinement costs')
