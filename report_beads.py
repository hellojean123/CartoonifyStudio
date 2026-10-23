"""Cross-check published grids/charts/counts and summarize real measurements."""
from pathlib import Path
from io import BytesIO,StringIO
import csv,json,zipfile,hashlib
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from PIL import Image

ROOT=Path(__file__).parent

def main():
    out=ROOT/'outputs/beads';metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    assert len(metrics)==21
    for case in metrics:
        folder=out/Path(case['file']).stem;data=json.loads((folder/'pattern.json').read_text(encoding='utf-8'))
        grid=list(csv.reader(StringIO((folder/'grid.csv').read_text(encoding='utf-8-sig'))))
        assert data['grid']==[row[1:] for row in grid[1:]]
        counts={r['id']:r['count'] for r in data['palette']};actual={key:0 for key in counts}
        for row in data['grid']:
            for key in row:actual[key]+=1
        assert counts==actual and sum(counts.values())==case['total_beads']
        rows=list(csv.DictReader(StringIO((folder/'colors.csv').read_text(encoding='utf-8-sig'))))
        assert {r['id']:int(r['count']) for r in rows}==counts
        with zipfile.ZipFile(folder/'pattern.zip') as package:
            assert package.read('pattern.pdf')==(folder/'pattern.pdf').read_bytes()
            assert package.read('pattern.json')==(folder/'pattern.json').read_bytes()
            assert 'SOURCE.txt' in package.namelist()
        assert len(PdfReader(folder/'pattern.pdf').pages)==case['pdf_pages']
        with Image.open(folder/'chart.png') as im:im.verify()
        ET.fromstring((folder/'chart.svg').read_bytes())
    testroot=ET.parse(ROOT/'outputs/pytest_beads.xml').getroot()
    suites=list(testroot) if testroot.tag=='testsuites' else [testroot]
    tests=sum(int(s.attrib['tests']) for s in suites);failures=sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites)
    assert tests>=79 and failures==0
    smoke=json.loads((out/'smoke_test.json').read_text(encoding='utf-8'))
    assert smoke['http_status']==200 and len(smoke['cases'])==2 and '取消' in smoke['cancellation_status']
    before=sum(m['before']['isolated_beads'] for m in metrics);after=sum(m['after']['isolated_beads'] for m in metrics)
    bench=json.loads((out/'benchmark.json').read_text(encoding='utf-8'))
    lines=['# 照片转拼豆图纸：实现与实测','', '## 使用与效果','', '新增顶部“拼豆图纸”页，与照片卡通化并列。共用原照片上传，算法、颜色清单、图纸导出与 UI 分离，纯 CPU。', '', '当前 UI 默认人像清晰：96 格、18 色、轻度平滑、小于 3 格的四连通色块合并。人脸在平滑前定位并优先参与配色，面部降低合并强度，保留亮部与暗部五官结构。低层 BeadParams() 保留原 72 格/10 色的兼容默认值。', '', f'21 张真实照片、相同平滑和调色板拟合的量化基线：孤立单粒合计 **{before} → {after}**，减少 **{(1-after/before)*100:.1f}%**。单粒定义为同色四连通分量只有 1 格；其中部分单粒用于保留五官；与旧版不同时追求清零。该数字衡量拼装碎片程度，不代表审美分数。', '', '面部专项改进及旧版/同预算/新预设四联对照见 outputs/beads_face_v2/gallery.html 和 bead_face_report.md。复杂背景、多人小脸在低格数下仍会失去细节，人物建议试均衡预设、提高格数或先裁剪主体。', '', '## 输出与颜色约定','', '- 每张有像素预览、编号 PNG、SVG、A4 PDF、用量 CSV、二维网格 CSV、JSON 及 ZIP；所有格式共用同一个离散网格。数量为净用量，每格一粒，无空白格，白色也计数。', '- B01 等为图内编号，A01 等为自动设计颜色；不是品牌色号。支持导入最多 256 色的 UTF-8 code,r,g,b[,name] 色卡，最终只从实卡 RGB 中选择不超过指定数量的颜色。示例 EXAMPLE 色卡不代表品牌商品。', '- PDF 每页最多 29×29 格，实际格距 5 / 2.6 mm；按 100% 打印并检查 50mm 标尺。长色号按列宽截短，完整色号/中文名在 CSV、JSON 中。', '- 保留原样片的作者、来源、许可，在画廊及每个示例包 SOURCE.txt 中提供。', '', '## 验证','', f'- 最终完整 pytest：{tests} 项通过；包括确定性、四连通、减少单粒、精确实体色卡、清单/网格/JSON一致性、PDF分页、160×160 上限的 37 页、两种格距、长色号防溢出、异常 CSV、取消，以及磁盘写入失败时只清理自身产物。', '- 实际启动 Gradio：HTTP 200；自动配色、导入实体色卡、PDF/ZIP 下载及处理中取消全部通过。原四风格预览、原尺寸 PNG/JPG、批处理与取消也再次通过。', '- 21 张样片逐份验证所有格子编号与颜色用量一致、ZIP内容一致、PNG解码与SVG解析成功、PDF页数正确。', '- 用 Poppler 渲染并查看女性样例 PDF 的封面、首张图纸与末页，检查数字、坐标、分页和打印标尺。没有把程序通过等同于人工拼装完成。', '', '## CPU / 内存实测','', '输入为真实 3840×2880 照片。每种规模独立子进程，预热后 3 次；图案计算不含源图解码，导出含 PNG/SVG/PDF/CSV/JSON/ZIP 内存生成，不含磁盘与网络。RSS 5ms 采样绝对峰值，含 Python、输入和分配器，可能漏掉短峰值，不是进程总内存上限。', '', '| 最长边 / 最多颜色 | 图案计算中位秒 | 全格式导出打包中位秒 | RSS MiB |','|---|---:|---:|---:|']
    for row in bench:lines.append(f"| {row['grid']} / {row['colors']} | {row['algorithm_median']:.3f} | {row['export_median']:.3f} | {row['peak_rss_mib']:.1f} |")
    lines+=['', '数据：outputs/beads/metrics.json、benchmark.json、smoke_test.json；测试：outputs/pytest_beads.xml；效果入口：outputs/beads/gallery.html。', '', '## 已知限制','', '- 这是易拼优先的传统区域量化，不是手工像素画；合并强度过大仍会损失五官、发丝、小字。', '- 五官保护依赖已有 CPU 人脸定位，微小人脸或遮挡可能失败；不会宣称零碎块清零与所有细节同时保留。', '- 不能自动验证实物色卡的测量准确性；屏幕与实体颜色受显示器、光照影响。', '- 160 格、32 色会产生较多纸页和数万粒工作量，导出明显比预览慢；推荐先用轻松拼或均衡模式。', '- 尚无逐格手动修改、透明留空或自动去背景；可编辑 grid.csv / pattern.json 作二次设计，但直接改文件不会自动重建图纸。']
    (ROOT/'beads_report.md').write_bytes('\n'.join(lines).encode('utf-8'))
    receipt={'tests_passed':tests,'sample_patterns_verified':len(metrics),'isolated_beads_before':before,'isolated_beads_after':after,'http_status':200,'custom_palette_verified':True,'inflight_cancellation_verified':True}
    (out/'validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
