"""Compare actual published v1 with new face handling at identical grid/color budgets."""
from pathlib import Path
from dataclasses import replace
import subprocess,sys,types,json,time
import cv2
import numpy as np
from PIL import Image
from cartoonify.core import load_image,resize
from cartoonify.beads import BeadParams,make_pattern,preview_image,export_files,package_files,portrait_maps
from evaluate import sheet
ROOT=Path(__file__).parent

def main():
    old=types.ModuleType('cartoonify.bead_baseline');old.__package__='cartoonify';sys.modules[old.__name__]=old
    code=subprocess.check_output(['git','show','463cd185fdd8f8074a7b66985d966c119de94cf0:cartoonify/beads.py'],cwd=ROOT).decode('utf-8')
    exec(compile(code,'bead_baseline','exec'),old.__dict__)
    out=ROOT/'outputs/beads_face_v2';out.mkdir(exist_ok=True);cards=[];rows=[]
    records=json.loads((ROOT/'samples/metadata.json').read_text(encoding='utf-8'))['images']
    for record in records:
        stem=Path(record['filename']).stem;im=load_image(ROOT/'samples'/record['filename'])
        before=old.make_pattern(im);same=make_pattern(im);recommended=make_pattern(im,BeadParams(96,18,1,3,1.5))
        panels=[im,old.preview_image(before),preview_image(same),preview_image(recommended)]
        sheet(panels,['Original','Published:72 / 10','New:same 72 / 10','Portrait:96 / 18'],out/(stem+'.jpg'),width=1800,height=650)
        h,w=same.labels.shape;face,features,_=portrait_maps(resize(im,1000),(w,h))
        region=features>.3
        reference=cv2.cvtColor(cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255,cv2.COLOR_RGB2LAB)
        def error(pattern):
            lab=cv2.cvtColor(pattern.palette[pattern.labels].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
            return float(np.linalg.norm(lab-reference,axis=2)[region].mean()) if region.any() else None
        rows.append({'file':record['filename'],'before':before.metrics,'same_budget':same.metrics,'portrait_preset':recommended.metrics,
                     'feature_deltaE76_before':error(before),'feature_deltaE76_after_same_budget':error(same)})
        cards.append(f'<article><h2>{stem}</h2><img loading="lazy" src="{stem}.jpg"></article>')
        print(stem,'faces',same.metrics['detected_faces'],'fragments',before.metrics['after']['isolated_beads'],'->',same.metrics['after']['isolated_beads'],flush=True)
    (out/'metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>拼豆面部细节改进</title><style>body{font:16px system-ui;max-width:1800px;margin:30px auto}img{width:100%}article{margin:30px}</style><h1>原图 / 已发布版 / 相同格数与颜色的新算法 / 人像预设</h1><p>同预算对照用于区分算法收益与增加格数、颜色的收益；没有自动审美评分。五官保护会有意保留少量小色块。</p><a href="../../samples/ATTRIBUTION.md">照片作者及授权</a>'+''.join(cards),encoding='utf-8')

if __name__=='__main__':main()
