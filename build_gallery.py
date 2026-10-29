"""Build a portable HTML gallery with individual source credits."""
import html
import json
from pathlib import Path
from cartoonify.core import STYLES

def main():
    root=Path(__file__).parent
    metadata=json.loads((root/'samples'/'metadata.json').read_text(encoding='utf-8'))
    cards=[]
    for r in metadata['images']:
        stem=Path(r['filename']).stem
        links=' · '.join(f'<a href="{i+1}_{style}/{stem}.png">{["动漫人像","扁平插画","夸张手绘漫画","铅笔素描"][i]} PNG</a>' for i,style in enumerate(STYLES))
        cards.append(f'<article><h2>{html.escape(r["filename"])}</h2><a href="{stem}_comparison.jpg"><img loading="lazy" src="{stem}_comparison.jpg"></a><p>{links}</p><details><summary>来源与署名</summary><p>{html.escape(r["title"])}</p><p>{html.escape(r["author"])} · {html.escape(r["license"])}</p><a href="{html.escape(r["source_url"],quote=True)}">Wikimedia Commons 原始来源</a></details></article>')
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Cartoonify Studio · 效果画廊</title><style>body{background:#171322;color:#eee;font:16px system-ui;max-width:1600px;margin:32px auto;padding:0 20px}img{width:100%;border-radius:12px}article{margin:32px 0;background:#242033;padding:18px;border-radius:18px}a{color:#c8adff}h2{font-size:18px}summary{cursor:pointer;color:#b9afcf}</style><h1>🎨 Cartoonify Studio</h1><p>原图 / 动漫人像 / 扁平插画 / 夸张手绘漫画 / 铅笔素描</p><p>21 张真实照片 · 16 张人物照片 · 84 张风格结果。点击对比图查看大图，点击下方链接查看各风格 PNG。</p><p><a href="../samples/ATTRIBUTION.md">完整署名与授权</a> · <a href="../benchmark_v4.md">v4 历史 CPU 实测</a></p>'+''.join(cards)+'</html>'
    extras='<p><a href="v3_review/gallery.html">历史 v3 与 v2 对照及局部</a> · <a href="../benchmark_v3.md">v3 CPU 实测</a></p><p><a href="blind_review.html">历史 v2 随机 A/B 盲评</a> · <a href="../upgrade_report.md">七项升级实测报告</a> · <a href="../samples/high_resolution/ATTRIBUTION.md">真实高清参考图署名</a></p><p>真实高清结果：'+' · '.join(f'<a href="highres_{i}.png">{style}</a>' for i,style in enumerate(STYLES,1))+'</p>'
    extras+='<p><a href="beads/gallery.html">拼豆图纸：更少碎块、可打印 PDF 与颜色用量</a> · <a href="../beads_report.md">拼豆实测报告</a></p>'
    extras+='<p><a href="beads_face_v2/gallery.html">拼豆人像：旧版 / 同预算改进 / 人像清晰预设</a></p>'
    extras='<p><a href="v4_review/gallery.html">历史 v4：原图 / 旧版 / v4 及五官局部</a> · <a href="../benchmark_v4.md">v4 历史 CPU 实测</a></p>'+extras
    extras+='<p><a href="pencil_v5/gallery.html">历史素描 v5：21 张新旧与 OpenCV 对照</a> · <a href="../pencil_v5_report.md">素描 CPU 实测与局限</a></p>'
    extras+='<p><a href="pencil_v6/gallery.html">当前素描 v6：变宽笔触与新旧对照</a> · <a href="pencil_v6/brush_sheet.png">单笔样张</a> · <a href="../pencil_v6_report.md">CPU 实测与局限</a></p>'
    extras+='<p><a href="anime_v7/gallery.html">当前动漫 v7：21 张动漫增强新旧对照</a> · <a href="../anime_v7_report.md">模型筛选、CPU 实测与局限</a></p>'
    extras+='<p><a href="flat_v8/gallery.html">当前扁平插画 v8：同预算和默认配色对照</a> · <a href="../flat_v8_report.md">实测与局限</a></p>'
    extras+='<p><a href="caricature_v9/gallery.html">夸张手绘漫画 v9：几何变形与四特征对照</a> · <a href="../caricature_v9_report.md">实测与局限</a></p>'
    document=document.replace('</html>',extras+'</html>')
    (root/'outputs'/'gallery.html').write_text(document,encoding='utf-8')

if __name__=='__main__': main()
