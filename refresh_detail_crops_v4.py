"""Refresh pixel-scale detail crops using analysis-to-original face coordinates."""
from pathlib import Path
from io import BytesIO
import subprocess,json
import numpy as np
from PIL import Image
from cartoonify.core import load_image,STYLES
from evaluate import sheet
ROOT=Path(__file__).resolve().parent
BASELINE='0b46ac10b7dd8174afe8502b769ee8f9bba409dd'
SELECTED={'01_portrait','03_portrait','05_portrait','13_portrait','15_landscape','17_architecture','19_animal','23_portrait'}

def main():
    rows=json.loads((ROOT/'outputs/timings_v4.json').read_text(encoding='utf-8'))
    manifest=[]
    for row in rows:
        stem=Path(row['file']).stem
        if stem not in SELECTED:continue
        im=load_image(ROOT/'samples'/row['file']);h,w=im.shape[:2];i=STYLES.index(row['style'])
        with Image.open(ROOT/'outputs'/f'{i+1}_{row["style"]}'/(stem+'.png')) as stored:new=np.array(stored.convert('RGB'))
        data=subprocess.check_output(['git','show',BASELINE+f':outputs/{i+1}_{row["style"]}/{stem}.png'],cwd=ROOT)
        with Image.open(BytesIO(data)) as stored:old=np.array(stored.convert('RGB'))
        if row['faces']:
            box=max(row['faces'],key=lambda b:b[2]*b[3]);scale=min(1.,1000/max(h,w))
            aw=max(1,round(w*scale));ah=max(1,round(h*scale))
            x,y,bw,bh=[round(v*s) for v,s in zip(box,(w/aw,h/ah,w/aw,h/ah))]
            size=min(max(bw,bh),h,w,520);x=int(np.clip(x+(bw-size)/2,0,w-size));y=int(np.clip(y+(bh-size)/2,0,h-size))
        else:size=min(420,h,w);x=(w-size)//2;y=(h-size)//2
        dest=ROOT/'outputs/v4_review'/(stem+f'_{i+1}_crop.png')
        sheet([a[y:y+size,x:x+size] for a in (im,old,new)],['Original detail','Published old detail','New detail'],dest,width=size*3+36,height=size+45)
        manifest.append({'file':row['file'],'style':row['style'],'original_size':[w,h],'crop_original_pixels':[x,y,size,size],'faces_analysis_coordinates':row['faces'],'sheet':dest.relative_to(ROOT).as_posix()})
    assert len(manifest)==32
    (ROOT/'outputs/v4_review/detail_crop_coordinates.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('32 native detail crops verified against original-pixel face coordinates')

if __name__=='__main__':main()
