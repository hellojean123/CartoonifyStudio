"""Render all licensed samples into real bead charts and measure fragmented regions."""
from dataclasses import replace
from pathlib import Path
import html,json,time,math
from io import BytesIO
from PIL import Image
from pypdf import PdfReader
from cartoonify.core import load_image
from cartoonify.beads import BeadParams,make_pattern,preview_image,export_files,package_files,palette_html
from evaluate import sheet

ROOT=Path(__file__).parent

def main():
    output=ROOT/'outputs/beads';output.mkdir(exist_ok=True)
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images'];rows=[];cards=[]
    for record in records:
        path=ROOT/'samples'/record['filename'];dest=output/path.stem;dest.mkdir(exist_ok=True)
        image=load_image(path);params=BeadParams();started=time.perf_counter()
        pattern=make_pattern(image,params);algorithm=time.perf_counter()-started
        baseline=make_pattern(image,replace(params,min_region=0,cohesion=0))
        started=time.perf_counter();files=export_files(pattern);export_seconds=time.perf_counter()-started
        files['SOURCE.txt']=(f"Source: {record['source_url']}\nAuthor: {record['author']}\nLicense: {record['license']}\nLicense URL: {record['license_url']}\nChanges: resized, palette quantized, spatial regions simplified, converted to bead charts.\nPhoto-derived pattern illustrations follow the source license above.\nTitle: {record['title']}\nPhoto derivatives retain the source license obligations.\n").encode('utf-8')
        for name,data in files.items():(dest/name).write_bytes(data)
        (dest/'pattern.zip').write_bytes(package_files(files))
        naive=preview_image(baseline);clean=preview_image(pattern)
        Image.fromarray(naive).save(dest/'quantized_only.png')
        sheet([image,naive,clean],['Original','Quantized only (same smoothing/palette fit)','Region simplified bead pattern'],dest/'comparison.jpg',height=580)
        pages=PdfReader(BytesIO(files['pattern.pdf'])).pages
        h,w=pattern.labels.shape;assert len(pages)==1+math.ceil(w/29)*math.ceil(h/29)
        assert sum(r['count'] for r in json.loads(files['pattern.json'])['palette'])==w*h
        row={'file':path.name,'algorithm_seconds':algorithm,'export_seconds':export_seconds,'pdf_pages':len(pages),**pattern.metrics};rows.append(row)
        cards.append(f'<article><h2>{html.escape(path.stem)}</h2><img loading="lazy" src="{path.stem}/comparison.jpg"><p>{w}×{h} 格 · {pattern.metrics["used_colors"]} 色 · {w*h} 粒 · 孤立单粒 {pattern.metrics["before"]["isolated_beads"]} → {pattern.metrics["after"]["isolated_beads"]}</p><p><a href="{path.stem}/pattern.pdf">可打印 PDF</a> · <a href="{path.stem}/chart.png">完整编号 PNG</a> · <a href="{path.stem}/chart.svg">矢量 SVG</a> · <a href="{path.stem}/pattern.zip">下载全部图纸与清单</a></p><details><summary>配色与净用量</summary>{palette_html(pattern)}</details><p>{html.escape(record["author"])} · {html.escape(record["license"])} · <a href="{html.escape(record["source_url"],quote=True)}">原照片来源</a></p></article>')
        print(path.name,'isolated',pattern.metrics['before']['isolated_beads'],'->',pattern.metrics['after']['isolated_beads'],round(algorithm,3),flush=True)
    (output/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cartoonify 拼豆图纸</title><style>body{font:16px system-ui;background:#f4f5fa;color:#172235;max-width:1500px;margin:30px auto;padding:0 20px}article{background:white;padding:20px;border-radius:16px;margin:24px 0}img{width:100%}a{color:#6940bd}summary{cursor:pointer}</style><h1>更容易拼的照片图纸</h1><p>原图 / 相同平滑与色板拟合的仅量化结果 / 区域聚合与小块合并。默认最长边 72 格，最多 10 色，小于 5 粒的色块优先合并。</p><p>自动颜色不是品牌色号。每格一粒；PDF 按实际大小 100% 打印，检查 50mm 标尺。简化会损失细节，人物可尝试均衡预设或降低小块阈值。</p>'+''.join(cards)+'</html>'
    (output/'gallery.html').write_text(document,encoding='utf-8')
    before=sum(r['before']['isolated_beads'] for r in rows);after=sum(r['after']['isolated_beads'] for r in rows)
    print('Verified',len(rows),'patterns/PDFs;',before,'->',after,'isolated beads',flush=True)

if __name__=='__main__':main()
