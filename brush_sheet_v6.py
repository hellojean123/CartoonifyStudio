"""Render standalone brush samples with the same production pencil brush."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from cartoonify.strokes import rasterize


def main():
    out=Path(__file__).resolve().parent/'outputs/pencil_v6';out.mkdir(exist_ok=True)
    rows=[('Fine / light, pressure varies',.28,.50,.9,0,1),
          ('Fine / dark, firm start + release',.32,.90,2.4,1,1),
          ('Medium, light start + firm end',.68,.85,1.25,2,1),
          ('Broad / light side stroke',1.3,.40,.65,4,1),
          ('Same fine stroke, 1 pass',.40,.65,1.2,0,1),
          ('Same fine stroke, 3 passes',.40,.65,1.2,0,3)]
    c=[];width=[];pressure=[];dose=[];profile=[];phase=[];length=[]
    for i,(_,w,p,d,kind,repeats) in enumerate(rows):
        y=48+i*70
        for _ in range(repeats):
            c.append([[30,y],[200,y-8],[400,y+9],[640,y]])
            width.append(w);pressure.append(p);dose.append(d);profile.append(kind);phase.append(.8);length.append(610)
    c=np.array(c,np.float32)
    plan={'control':c,'width':np.array(width,np.float32),'pressure':np.array(pressure,np.float32),
          'dose':np.array(dose,np.float32),'profile':np.array(profile,np.float32),'phase':np.array(phase,np.float32),
          'length':np.array(length,np.float32),'shape':np.array([440,680],np.int32),
          'bounds':np.concatenate((c.min(1),c.max(1)),1)}
    field=rasterize(plan,(880,1360));canvas=Image.fromarray((255*np.exp(-field)).astype(np.uint8)).convert('RGB')
    draw=ImageDraw.Draw(canvas)
    for i,(label,*_) in enumerate(rows):draw.text((55,34+i*140),label,fill=(40,40,40))
    canvas.save(out/'brush_sheet.png')
    print('Single-stroke brush sheet rendered with production brush at 2x reference scale',flush=True)


if __name__=='__main__':main()
