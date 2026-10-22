"""Audit actual deliverables and generate evidence-based upgrade reports."""
from pathlib import Path
import json
import statistics
import xml.etree.ElementTree as ET
from PIL import Image
from cartoonify.core import STYLES

ROOT=Path(__file__).parent

def main():
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    assert len(records)>=20 and sum(r['category']=='portrait' for r in records)>=15
    count=0
    for record in records:
        original=Image.open(ROOT/'samples'/record['filename'])
        for i,style in enumerate(STYLES):
            path=ROOT/'outputs'/f'{i+1}_{style}'/(Path(record['filename']).stem+'.png')
            with Image.open(path) as image:
                assert image.format=='PNG' and image.mode=='RGB' and image.size==original.size
                image.verify()
            count+=1
        assert (ROOT/'outputs'/(Path(record['filename']).stem+'_comparison.jpg')).exists()
    native=Image.open(ROOT/'samples/high_resolution/reference_4k.jpg')
    assert native.size==(3840,2880)
    for i in range(1,5):
        with Image.open(ROOT/f'outputs/highres_{i}.png') as image: assert image.size==native.size;image.verify()
        assert (ROOT/f'outputs/upgrade_comparisons/real4k_{i}.jpg').exists()
    tests=ET.parse(ROOT/'outputs/pytest.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests'])>=50 and tests.attrib['failures']=='0' and tests.attrib['errors']=='0'
    smoke=json.loads((ROOT/'outputs/smoke_test.json').read_text(encoding='utf-8'))
    assert smoke['http_status']==200 and smoke['full_png'] and smoke['batch_png_count']==2
    assert len(smoke['exports'])==4 and all(r['preview_stages']>=2 for r in smoke['exports'])
    assert '取消 True' in smoke['cancelled_batch_status']
    benchmarks=json.loads((ROOT/'outputs/benchmark_v2.json').read_text(encoding='utf-8'))
    assert len(benchmarks)==32 and all(len(r['runs'])==3 for r in benchmarks)
    quant=json.loads((ROOT/'outputs/quantization_benchmark.json').read_text(encoding='utf-8'))
    timings=json.loads((ROOT/'outputs/timings.json').read_text(encoding='utf-8'))
    assert len(timings)==count
    blind=json.loads((ROOT/'outputs/blind_key.json').read_text(encoding='utf-8'))
    assert len(blind)==24
    for r in blind: assert (ROOT/'outputs/blind'/(r['id']+'.jpg')).exists()
    rows=[]
    for style in STYLES:
        values=[r['preview_export_mae_0_255'] for r in timings if r['style']==style]
        rows.append(f'| {style} | {statistics.mean(values):.3f} | {max(values):.3f} |')
    means='\n'.join(rows)
    quant_speed=quant[0]['median_seconds']/quant[1]['median_seconds']
    v4k=[r for r in benchmarks if r['version']=='v2' and r['width']==3840]
    numbers='\n'.join(f'| {r["style"]} | {r["median_seconds"]:.3f} | {r["cached_parameter_median"]:.3f} | {r["cached_preview_median"]:.3f} | {r["rss_peak_mib"]:.1f} |' for r in v4k)
    face_count=sum(bool(r['faces']) for r in timings if r['style']==STYLES[0] and 'portrait' in r['file'])
    report=f'''# 七项升级完成与实际评估

## 交付状态

| 升级项 | 已实现 | 实际证据 |
|---|---|---|
| 高清与预览尺度一致 | 同一分析图和色板；输出尺度网点；原图引导系数重建、细线重算、带重叠分块 | 尺度误差与接缝测试；4 张真实 4K 输出、原生裁剪 |
| 人像保护与轮廓分层 | 离线 Haar 检测、羽化保护、手动百分比保护框；粗细线分开 | 真实人像自动检测、人工保护与参数回归 |
| 资源队列与无效参数 | 共用 cpu 并发组和资源锁；风格专属范围/控件；取消 | 配置测试、资源串行测试、HTTP 运行中取消 |
| 插画区域约束与量化加速 | Mean Shift、Lab 聚类、32³ LUT、局部空间投票、脸部连续渐变 | 旧新量化独立计时、去孤立块合成测试、人物对照图 |
| 缓存与渐进预览 | 64 MiB 解码 + 112 MiB 中间缓存；独立脸框缓存；480 草稿→1000 最终预览 | 字节限制/失效/只读/线程测试；四风格 API 均返回至少两阶段 |
| 漫画与素描 | 网点尺度/角度/浓度与抗锯齿；XDoG 软阈值线、排线、交叉线与纸粒 | 新增参数实际改变输出的测试，前后对照与盲评 |
| 效果与性能评估 | 全量逐图指标、随机 A/B 投票、真实高清参考、重复计时和独立进程 RSS | 84 PNG、21 五联图、24 盲评题、32 性能案例及原始 JSON |

## 实际测试

pytest：{tests.attrib['tests']} 项全部通过，无错误或失败。见 outputs/pytest.xml。

Web 服务实际 HTTP 200；四种风格均完成草稿和最终预览；原尺寸 JPG/PNG 正常；ZIP 确认两张结果；运行中取消返回：{smoke['cancelled_batch_status']}。

已逐张打开核验 {count} 张正式 PNG 的格式、RGB 模式和源图尺寸；额外 4 张真实 3840×2880 输出同样验证。高清源图与授权见 samples/high_resolution。

## 性能实测

独立量化（1000 边长、16 色、3 次中位数）：旧版 {quant[0]['median_seconds']:.3f} 秒，新版 {quant[1]['median_seconds']:.3f} 秒，旧/新时间比 {quant_speed:.2f}。新算法的颜色和空间约束不同，不是逐像素等价替换。

| 4K 风格 | 无中间缓存中位秒 | 调轮廓高清缓存中位秒 | 最终预览缓存中位秒 | 进程峰值 RSS MiB |
|---|---:|---:|---:|---:|
{numbers}

详细 640/720p/1080p/4K 新旧对照、P95、编码、解码和分阶段时间见 benchmark.md 与 outputs/benchmark_v2.json。只有 3 次的百分位仅作描述，不代表统计置信度。缓存与高清重建承担不同工作，不能承诺所有场景比旧版快。

## 最终预览与精细导出一致性

对 21 张照片逐风格比较：把精细结果缩回 1000，与最终标准预览计算 RGB 平均绝对误差（0～255）。误差不等于艺术质量，精细新增细节允许有差别。

| 风格 | 全集平均误差 | 全集最大误差 |
|---|---:|---:|
{means}

所有原始指标见 outputs/timings.json。细碎边缘比例只作诊断，不能将线条越少直接解释为效果越好。

## 人像与目视复核

16 张人物图中，经典预设在 {face_count} 张检测到至少一个候选人脸；这不是带人工真值的检测准确率。Haar 可能漏检侧脸、小脸和遮挡，已提供手动保护框。

实际查看老人、女性、儿童、多人及动物对比。新版插画更连续，人物局部减少碎块；经典卡通更平滑；漫画保留可调网点；素描首轮偏淡，已改进结构线并重新生成。纸纹与排线属于程序化效果，仍有审美偏好差异。

outputs/blind_review.html 提供 24 道固定随机 A/B 对比，可独立投票并导出；程序没有代替用户打分或宣称盲评获胜。真实 4K 原生局部图位于 outputs/upgrade_comparisons/real4k_*.jpg。

## 兼容性与限制

仍为 Python 3.12、OpenCV、NumPy、Pillow、Gradio、pytest，未增加深度模型、GPU 或付费 API。OpenCL 显式关闭；已支持 ICC→sRGB、EXIF、透明度与含 sRGB 配置的 PNG/JPG 导出。

精细导出使用多尺度抽象和原图引导重建，不是生成式重绘，也不是所有滤波都在原始分辨率执行。脸部区域并非精确皮肤分割；强阴影/复杂背景仍可能有线条；网点缩小可能出现摩尔纹；缓存上限不等于总进程内存上限。输入 1600 万像素、40 MB 上限保留。

最小旧版算法参考位于 benchmarks/legacy_v1.py，清理整套本地备份不影响评估复现。升级结果与此报告仅根据本次实际输出和测试生成。
'''
    (ROOT/'upgrade_report.md').write_text(report,encoding='utf-8')
    (ROOT/'test_report.md').write_text('# v2 测试与交付核验\n\n'+f'pytest {tests.attrib["tests"]} 项全部通过；{len(records)} 张输入、{count} 张风格 PNG、21 张对比、24 道 A/B 盲评、4 张真实 4K 均核验。\n\nWeb HTTP、双阶段预览、原尺寸 JPG/PNG、ZIP 和运行中取消均通过。\n\n完整项目升级、性能与限制见 upgrade_report.md；原始证据见 outputs/pytest.xml、smoke_test.json、benchmark_v2.json、timings.json。\n',encoding='utf-8')
    print(f'Validated {tests.attrib["tests"]} tests, {count} outputs + 4 real-4K, 24 blind comparisons, HTTP exports and cancellation',flush=True)

if __name__=='__main__': main()
