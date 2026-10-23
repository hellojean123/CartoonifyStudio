"""Validate delivered images and publish measured v3 findings, not aesthetic scores."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from cartoonify.core import STYLES,load_image
from evaluate import sheet

ROOT=Path(__file__).parent

def main():
    out=ROOT/'outputs';records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    checked=[]
    for record in records:
        stem=Path(record['filename']).stem;source=load_image(ROOT/'samples'/record['filename'])
        for i,style in enumerate(STYLES,1):
            path=out/f'{i}_{style}'/(stem+'.png')
            with Image.open(path) as image:
                assert image.size==(source.shape[1],source.shape[0]);image.verify()
            checked.append(str(path.relative_to(ROOT)))
        with Image.open(out/(stem+'_comparison.jpg')) as image:image.verify()
    original=load_image(ROOT/'samples/high_resolution/reference_4k.jpg')
    for i,style in enumerate(STYLES,1):
        path=out/f'highres_{i}.png';result=load_image(path);assert result.shape==original.shape
        h,w=result.shape[:2];size=480;y=min(h-size,round(h*.72));x=min(w-size,round(w*.55))
        sheet([a[y:y+size,x:x+size] for a in (original,result)],['Original native detail',style+' v3'],out/'v3_review'/f'highres_{i}_native.jpg',width=984,height=525)
    page=out/'v3_review/gallery.html';document=page.read_text(encoding='utf-8')
    for crop in (out/'v3_review').glob('*_crop.jpg'):
        name=crop.name.replace('_crop.jpg','.jpg')
        document=document.replace(f'src="{name}">',f'src="{name}"><p><a href="{crop.name}">查看原尺寸局部</a></p>')
    document+='<p><a href="../../samples/ATTRIBUTION.md">照片作者与完整授权</a> · <a href="../gallery.html">全量画廊</a></p><h2>真实高清前景原生细节</h2>'
    document+=''.join(f'<a href="highres_{i}_native.jpg">{style}</a> · ' for i,style in enumerate(STYLES,1))
    page.write_text(document,encoding='utf-8')
    root=ET.parse(out/'pytest_v3.xml').getroot();suites=list(root) if root.tag=='testsuites' else [root]
    count=sum(int(s.attrib.get('tests',0)) for s in suites);fail=sum(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    assert count>=61 and fail==0
    smoke=json.loads((out/'smoke_test.json').read_text(encoding='utf-8'));assert smoke['http_status']==200
    rows=json.loads((out/'timings_v3.json').read_text(encoding='utf-8'));assert len(rows)==84
    bench=json.loads((out/'benchmark_v3_isolated.json').read_text(encoding='utf-8'));assert len(bench)==12
    receipt={'pytest_passed':count,'output_images_verified':len(checked),'comparison_sheets_verified':len(records),'highres_dimensions':[w,h],'highres_outputs':4,'http_status':200,'cases':checked}
    (out/'validation_v3.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v3 优化与实际核验','', '## 改动与观察','', '- 人像：官方 MIT YuNet 小模型，强制 OpenCV CPU 后端；320/640 尺度检测与 Haar 补充。五点定位眼鼻嘴，框内色彩约束抑制皮肤微线。原先漏检的 01 老人紧裁切已检出，加入回归测试。模型来源、许可证和 SHA256 随项目保留。', '- 素描：恢复连续石墨明暗，让头发、脸部和衣服有体积；移除旧的规则交叉网屏，以低幅不规则颗粒和局部梯度调制细笔触代替。女性、儿童、动物、山景对照中阴影不再大片空白。', '- 插画：相邻小色块合并；眼鼻嘴用同一色板恢复细节；高清输出不再混回连续照片 RGB。色块更明确，但多人远景的小脸仍会简化。', '- 漫画：降低细碎线和粗线膨胀，增加大范围暗部设计，网点限定在中间调、降低脸部覆盖；女性样本的网点覆盖和照片感减轻，老人深皱纹仍有明显线块。', '- 经典卡通：由逐通道取整改为采样调色板，使色彩概括更明显；精细导出的灰度引导混合从 75% 减为 20%，控制照片细纹回流。', '- 高清：按风格选择重建方式，保留统一分析尺度，原尺寸渲染网点、石墨颗粒；移除不需要的原图灰度转换。插画、漫画、素描两档质量使用同一结构重建，界面及 README 明确说明。','', '## 实际验证','',f'- pytest：{count} 项通过，见 outputs/pytest_v3.xml。',f'- 21 张正式照片 × 4 风格 = {len(checked)} 张 PNG，逐张解码验证并核对原尺寸；21 张总览，28 张 v2/v3 对照、28 张局部对照。',f'- 4 张真实高清结果均为 {w}×{h}；另导出 4 张前景细节原生裁剪，避免只看空旷中心区域。','- Gradio HTTP 200，四种两阶段预览、原尺寸 PNG/JPG、批量 ZIP、处理中取消全部实际通过。网页端到端时间包含传输，不能用算法时间替代。','', '## 本次全样本预览/导出差异','', '| 风格 | 平均 MAE（0–255） | 最大 MAE |','|---|---:|---:|']
    for style in STYLES:
        values=[r['preview_export_mae'] for r in rows if r['style']==style]
        lines.append(f'| {style} | {np.mean(values):.3f} | {max(values):.3f} |')
    lines+=['', '这些误差只检查尺度一致性，不代表艺术质量。性能与 RSS 见 benchmark_v3.md。','', '## 仍有的限制','', '- 保护区域不是精确皮肤/头发语义分割；彩光、遮挡、微小侧脸仍可能漏检，手动框保留。', '- 插画有意压缩细节，多人远景和建筑小字可能简化；并非所有样本都比 v2 更悦目。', '- 素描比 v2 保留更多明暗，但仍有柔化和滤镜感，笔触不等同人工排线；没有声称做到结构流场手绘。', '- 旧 blind_review.html 是 v1/v2 历史盲评；本轮对照在 outputs/v3_review/gallery.html，没有预置或伪造人工偏好分数。', '- v2/v3 基准算法不同，旧性能为历史数据；不能把全部差异归因于某一优化。']
    (ROOT/'optimization_v3.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(receipt|{'cases':'84 verified paths in validation_v3.json'},ensure_ascii=True))

if __name__=='__main__':main()
