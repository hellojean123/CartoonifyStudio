"""CPU bead patterns: discrete cells, spatial simplification and physical color lists.

No dithering and no invented manufacturer codes. Algorithm is independent of UI.
"""
from dataclasses import dataclass
from io import StringIO, BytesIO
import csv
import html
import json
import math
import threading
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .core import MAX_PIXELS, resize

_CLUSTER_LOCK=threading.Lock()
PRESETS={'轻松拼（推荐）':(72,10,3,5,2.5),'均衡':(96,16,2,3,1.5),'保留细节':(128,24,1,1,.5)}

@dataclass(frozen=True)
class BeadParams:
    grid: int=72
    colors: int=10
    smooth: int=3
    min_region: int=5
    cohesion: float=2.5
    pitch_mm: float=5.
    preserve_features: bool=True

@dataclass
class BeadPattern:
    labels: np.ndarray
    palette: np.ndarray
    codes: list[str]
    names: list[str]
    metrics: dict
    pitch_mm: float

def _cancel(cancel):
    if cancel and cancel(): raise InterruptedError('拼豆生成已取消')

def normalized(params):
    p=params
    for value in (p.grid,p.colors,p.smooth,p.min_region,p.cohesion,p.pitch_mm):
        if not math.isfinite(float(value)): raise ValueError('拼豆参数必须是有限数字')
    return BeadParams(int(np.clip(p.grid,16,160)),int(np.clip(p.colors,2,32)),int(np.clip(p.smooth,0,4)),
                      int(np.clip(p.min_region,0,12)),float(np.clip(p.cohesion,0,4)),float(np.clip(p.pitch_mm,2.6,5)),bool(p.preserve_features))

def read_palette(path):
    """CSV columns code,r,g,b[,name], <=256 entries; explicit supplied colors only."""
    from pathlib import Path
    path=Path(path)
    if path.stat().st_size>128*1024: raise ValueError('色卡 CSV 不能超过 128 KB')
    try:
        reader=csv.DictReader(StringIO(path.read_text(encoding='utf-8-sig')))
        if not {'code','r','g','b'}.issubset(reader.fieldnames or []): raise ValueError('需要 code,r,g,b 列，可加 name 列')
        colors=[];codes=[];names=[]
        for row in reader:
            if len(colors)>=256: raise ValueError('色卡最多 256 色')
            code=row['code'].strip()
            if not code or len(code)>48 or code in codes: raise ValueError('色号须唯一、非空，最多 48 字符')
            rgb=tuple(int(row[c]) for c in ('r','g','b'))
            if any(v<0 or v>255 for v in rgb): raise ValueError('RGB 必须是 0～255 的整数')
            colors.append(rgb);codes.append(code);names.append((row.get('name') or code).strip()[:80])
        if not colors: raise ValueError('色卡不能为空')
        return np.array(colors,np.uint8),codes,names
    except (UnicodeError,TypeError,KeyError) as exc: raise ValueError('无法读取色卡，请使用 UTF-8 CSV 和整数 RGB') from exc

def _nearest(lab,centers):
    distance=((lab[...,None,:]-centers)**2).sum(axis=-1)
    return np.argmin(distance,axis=-1).astype(np.uint8),distance

def _spatial_labels(lab,centers,cohesion,cancel):
    labels,data=_nearest(lab,centers)
    if not cohesion:return labels
    # Red-black ICM avoids synchronous label swapping on checkerboard noise.
    yy,xx=np.indices(labels.shape)
    penalty=cohesion*190.
    for _ in range(4):
        _cancel(cancel)
        for parity in (0,1):
            padded=np.pad(labels,1,mode='edge');cost=data.copy()
            for dy,dx in ((0,1),(2,1),(1,0),(1,2)):
                neighbor=padded[dy:dy+labels.shape[0],dx:dx+labels.shape[1]]
                cost+=penalty*(np.arange(len(centers))[None,None,:]!=neighbor[...,None])
            winner=cost.argmin(axis=-1).astype(np.uint8)
            selected=(yy+xx)%2==parity;labels[selected]=winner[selected]
    return labels

def cleanup_regions(labels,centers,min_region,cancel=None,protected=None):
    """Merge small 4-connected islands into an adjacent color, never a new color."""
    result=labels.copy()
    if min_region<2:return result
    for _ in range(4):
        changed=False
        for color in range(len(centers)):
            _cancel(cancel)
            count,components,stats,_=cv2.connectedComponentsWithStats((result==color).astype(np.uint8),connectivity=4)
            for component in range(1,count):
                x,y,w,h,area=stats[component]
                if area>=min_region:continue
                x0=max(0,x-1);y0=max(0,y-1);x1=min(result.shape[1],x+w+1);y1=min(result.shape[0],y+h+1)
                view=result[y0:y1,x0:x1];island=(components[y0:y1,x0:x1]==component)&(view==color)
                if not island.any():continue
                if protected is not None and np.any(protected[y0:y1,x0:x1]&island):continue
                border=cv2.dilate(island.astype(np.uint8),np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)).astype(bool)&~island
                neighbors=view[border];neighbors=neighbors[neighbors!=color]
                if not neighbors.size:continue
                candidates,contacts=np.unique(neighbors,return_counts=True)
                distance=np.linalg.norm(centers[candidates]-centers[color],axis=1)
                # Prefer a long shared border; chroma is a secondary merge criterion.
                score=distance*.25-contacts.astype(float)*8
                view[island]=candidates[score.argmin()];changed=True
        if not changed:break
    return result

def _feature_cells(work,labels,centers):
    """Preserve dark eye/lip cells only, rather than every texture within a face."""
    from .core import face_mask
    _,_,landmarks=face_mask(work,details=True)
    h,w=labels.shape;scale=np.array([w/work.shape[1],h/work.shape[0]])
    protected=np.zeros((h,w),bool);light=centers[labels,0]
    for box,points in landmarks:
        bw=box[2]*scale[0];bh=box[3]*scale[1]
        if bw<6:continue
        for index in (0,1,3,4):
            px,py=points[index]*scale
            rx=max(1,round(bw*.12));ry=max(1,round(bh*.065))
            x0=max(0,round(px)-rx);x1=min(w,round(px)+rx+1)
            y0=max(0,round(py)-ry);y1=min(h,round(py)+ry+1)
            patch=light[y0:y1,x0:x1]
            if patch.size and patch.max()-patch.min()>3:
                cutoff=float(np.percentile(patch,35))
                protected[y0:y1,x0:x1]|=patch<=cutoff
    return protected

def region_metrics(labels,threshold=5):
    single=small=regions=0
    for value in np.unique(labels):
        n,_,stats,_=cv2.connectedComponentsWithStats((labels==value).astype(np.uint8),connectivity=4)
        areas=stats[1:,cv2.CC_STAT_AREA];regions+=n-1;single+=int(np.sum(areas==1));small+=int(np.sum(areas<max(2,threshold)))
    return {'regions':regions,'isolated_beads':single,'small_regions':small}

def make_pattern(image,params=None,custom_palette=None,cancel=None):
    p=normalized(params or BeadParams());im=np.asarray(image)
    if im.ndim==2 and im.size:im=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    if im.ndim!=3 or im.shape[2]!=3 or im.dtype!=np.uint8 or not im.size:raise ValueError('需要非空 uint8 RGB 照片')
    h,w=im.shape[:2]
    if h*w>MAX_PIXELS or max(h,w)>20000:raise ValueError('照片超过处理上限')
    _cancel(cancel);shape=(max(1,round(w*p.grid/max(h,w))),max(1,round(h*p.grid/max(h,w))))
    work=resize(im,1000)
    if p.smooth:work=cv2.bilateralFilter(work,7,45+p.smooth*15,6+p.smooth*4)
    cells=cv2.resize(work,shape,interpolation=cv2.INTER_AREA)
    if p.smooth:cells=cv2.bilateralFilter(cells,5,35+p.smooth*13,.7+p.smooth*.4)
    lab=cv2.cvtColor(cells.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    points=lab.reshape(-1,3);sample=points[np.linspace(0,len(points)-1,min(6000,len(points)),dtype=int)]
    k=min(p.colors,len(np.unique(sample,axis=0)))
    if k==1:centers=sample.mean(axis=0,keepdims=True)
    else:
        with _CLUSTER_LOCK:
            cv2.setRNGSeed(321)
            _,_,centers=cv2.kmeans(sample,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.4),1,cv2.KMEANS_PP_CENTERS)
    _cancel(cancel)
    if custom_palette is None:
        palette=np.clip(cv2.cvtColor(centers[None,:,:],cv2.COLOR_LAB2RGB)[0]*255,0,255).round().astype(np.uint8)
        codes=[f'A{i+1:02d}' for i in range(len(palette))];names=['自动配色']*len(palette)
    else:
        rgb,codes_all,names_all=custom_palette
        rgb=np.asarray(rgb)
        if rgb.ndim!=2 or rgb.shape[1]!=3 or not 1<=len(rgb)<=256 or rgb.dtype!=np.uint8 or len(codes_all)!=len(rgb) or len(names_all)!=len(rgb):raise ValueError('实体色卡格式无效')
        available=cv2.cvtColor(rgb[None,:,:].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[0]
        chosen=np.unique(_nearest(centers,available)[0])
        palette=rgb[chosen];codes=[codes_all[i] for i in chosen];names=[names_all[i] for i in chosen]
    centers=cv2.cvtColor(palette[None,:,:].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[0]
    raw,_=_nearest(lab,centers)
    labels=_spatial_labels(lab,centers,p.cohesion,cancel)
    protected=_feature_cells(work,raw,centers) if p.preserve_features else np.zeros(raw.shape,bool)
    labels[protected]=raw[protected]
    labels=cleanup_regions(labels,centers,p.min_region,cancel,protected)
    # Compact and order by usage, so B01 always denotes the most-used actual color.
    used,counts=np.unique(labels,return_counts=True);used=used[np.argsort(-counts,kind='stable')]
    remap=np.zeros(len(palette),np.uint8);remap[used]=np.arange(len(used),dtype=np.uint8)
    result=remap[labels]
    metrics={'before':region_metrics(raw,p.min_region),'after':region_metrics(result,p.min_region),
             'requested_colors':p.colors,'used_colors':len(used),'width':shape[0],'height':shape[1],
             'total_beads':int(result.size),'custom_palette':custom_palette is not None,'min_region':p.min_region,
             'protected_feature_cells':int(protected.sum())}
    return BeadPattern(result,palette[used],[codes[i] for i in used],[names[i] for i in used],metrics,p.pitch_mm)

def color_rows(pattern):
    counts=np.bincount(pattern.labels.ravel(),minlength=len(pattern.palette))
    return [{'id':f'B{i+1:02d}','code':pattern.codes[i],'name':pattern.names[i],
             'hex':'#'+''.join(f'{int(v):02X}' for v in rgb),'rgb':[int(v) for v in rgb],'count':int(counts[i])}
            for i,rgb in enumerate(pattern.palette)]

def preview_image(pattern):
    rgb=pattern.palette[pattern.labels];scale=max(1,min(12,900//max(pattern.labels.shape)))
    return np.repeat(np.repeat(rgb,scale,axis=0),scale,axis=1)

def _font(size):
    from pathlib import Path
    path=Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)

def chart_png(pattern,cancel=None):
    h,w=pattern.labels.shape;cell=24;left=52;top=72;rows=color_rows(pattern)
    width=max(560,left+w*cell+28);height=top+h*cell+70+math.ceil(len(rows)/4)*38
    image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image);font=_font(10);title=_font(16)
    draw.text((16,12),f'Cartoonify Beads | {w} x {h} | {len(rows)} colors | {h*w} beads',font=title,fill='#172235')
    draw.text((16,35),'B01... are local IDs. Match physical bead codes with colors.csv.',font=font,fill='#465064')
    for y in range(h):
        _cancel(cancel)
        for x in range(w):
            index=int(pattern.labels[y,x]);rgb=tuple(map(int,pattern.palette[index]));x0=left+x*cell;y0=top+y*cell
            draw.rectangle((x0,y0,x0+cell,y0+cell),fill=rgb)
            color='white' if np.dot(rgb,[.299,.587,.114])<135 else '#111111'
            draw.text((x0+cell/2,y0+cell/2),f'{index+1:02d}',anchor='mm',font=font,fill=color)
    for x in range(w+1):draw.line((left+x*cell,top,left+x*cell,top+h*cell),fill='#222222' if x%5==0 else '#999999',width=2 if x%5==0 else 1)
    for y in range(h+1):draw.line((left,top+y*cell,left+w*cell,top+y*cell),fill='#222222' if y%5==0 else '#999999',width=2 if y%5==0 else 1)
    for x in range(w):draw.text((left+(x+.5)*cell,top-15),str(x+1),anchor='mm',font=font,fill='black')
    for y in range(h):draw.text((left-16,top+(y+.5)*cell),str(y+1),anchor='mm',font=font,fill='black')
    for i,row in enumerate(rows):
        x=16+(i%4)*(width//4);y=top+h*cell+26+(i//4)*38
        draw.rectangle((x,y,x+18,y+18),fill=row['hex'],outline='#333333')
        draw.text((x+24,y+3),f"{row['id']} {row['count']} beads",font=font,fill='black')
    output=BytesIO();image.save(output,format='PNG',dpi=(cell*25.4/pattern.pitch_mm,)*2)
    return output.getvalue()

def chart_svg(pattern,cancel=None):
    h,w=pattern.labels.shape;cell=24;left=52;top=64;width=max(560,w*cell+80);height=top+h*cell+50
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif">',f'<text x="16" y="24" font-size="16">Bead chart: {w} x {h}. IDs match colors.csv.</text>']
    for y in range(h):
        _cancel(cancel)
        for x in range(w):
            i=int(pattern.labels[y,x]);rgb=pattern.palette[i];color='#'+''.join(f'{v:02x}' for v in rgb);ink='white' if np.dot(rgb,[.299,.587,.114])<135 else 'black'
            parts.append(f'<rect x="{left+x*cell}" y="{top+y*cell}" width="24" height="24" fill="{color}" stroke="#888" stroke-width=".4"/><text x="{left+(x+.5)*cell}" y="{top+(y+.5)*cell+3.5}" text-anchor="middle" font-size="10" fill="{ink}">{i+1:02d}</text>')
    for x in range(w):parts.append(f'<text x="{left+(x+.5)*cell}" y="{top-12}" text-anchor="middle" font-size="10">{x+1}</text>')
    for y in range(h):parts.append(f'<text x="{left-15}" y="{top+(y+.5)*cell+3}" text-anchor="middle" font-size="10">{y+1}</text>')
    for x in range(0,w+1,5):parts.append(f'<path d="M{left+x*cell} {top}V{top+h*cell}" stroke="#222" stroke-width="1.4"/>')
    for y in range(0,h+1,5):parts.append(f'<path d="M{left} {top+y*cell}H{left+w*cell}" stroke="#222" stroke-width="1.4"/>')
    return (''.join(parts)+'</g></svg>').encode('utf-8')

def pattern_pdf(pattern,cancel=None):
    """Vector A4 pages, actual specified cell pitch; never scale to fit in printing."""
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    output=BytesIO();canvas=Canvas(output,pagesize=A4,pageCompression=1,invariant=1)
    canvas.setTitle('Cartoonify Studio - Bead Pattern');h,w=pattern.labels.shape;rows=color_rows(pattern)
    canvas.setFont('Helvetica-Bold',22);canvas.drawString(42,792,'Cartoonify Beads')
    canvas.setFont('Helvetica',11);canvas.drawString(42,768,f'{w} columns x {h} rows | {len(rows)} colors | {h*w} beads')
    canvas.drawString(42,750,f'Print at 100% / actual size. Cell pitch: {pattern.pitch_mm:g} mm.')
    canvas.setFont('Helvetica',9);canvas.drawString(42,734,'B01... are local IDs, not manufacturer codes. Exact codes/RGB/counts: colors.csv.')
    thumb=Image.fromarray(pattern.palette[pattern.labels]);thumb=thumb.resize((w*4,h*4),Image.Resampling.NEAREST)
    scale=min(220/thumb.width,210/thumb.height);canvas.drawImage(ImageReader(thumb),42,498,width=thumb.width*scale,height=thumb.height*scale)
    canvas.setFont('Helvetica',9);canvas.drawString(42,478,'Color inventory (code column is ASCII; use CSV for full Unicode names):')
    for i,row in enumerate(rows):
        col=i//16;line=i%16;x=42+col*270;y=452-line*23
        canvas.setFillColorRGB(*(v/255 for v in row['rgb']));canvas.rect(x,y-3,15,15,fill=1,stroke=0)
        canvas.setFillColorRGB(.08,.1,.14);canvas.setFont('Helvetica',9)
        original_code=row['code'].encode('ascii','replace').decode();length=len(original_code)
        while True:
            code=original_code[:length]+('...' if length<len(original_code) else '')
            label=f"{row['id']}  {code}  {row['count']} beads"
            if canvas.stringWidth(label,'Helvetica',9)<=230 or length<=1:break
            length-=1
        canvas.drawString(x+23,y,label)
    canvas.setFont('Helvetica',9);canvas.drawString(42,50,'The image is intentionally simplified. Match bead colors using your physical color card.')
    canvas.showPage();cell=pattern.pitch_mm*mm;board=29;pages=math.ceil(w/board)*math.ceil(h/board);page=0
    for y0 in range(0,h,board):
        for x0 in range(0,w,board):
            _cancel(cancel);page+=1;bh=min(board,h-y0);bw=min(board,w-x0);left=78;top=730
            canvas.setFillColorRGB(.08,.1,.14);canvas.setFont('Helvetica-Bold',14)
            canvas.drawString(42,792,f'Bead chart | tile {page}/{pages}')
            canvas.setFont('Helvetica',10);canvas.drawString(42,771,f'Columns {x0+1}-{x0+bw} | Rows {y0+1}-{y0+bh} | {pattern.pitch_mm:g} mm cells')
            for y in range(bh):
                for x in range(bw):
                    index=int(pattern.labels[y0+y,x0+x]);rgb=pattern.palette[index]/255
                    canvas.setFillColorRGB(*map(float,rgb));canvas.rect(left+x*cell,top-(y+1)*cell,cell,cell,fill=1,stroke=0)
                    ink=1. if np.dot(rgb,[.299,.587,.114])<.53 else 0.
                    canvas.setFillColorRGB(ink,ink,ink);canvas.setFont('Helvetica',min(7,cell*.44))
                    canvas.drawCentredString(left+(x+.5)*cell,top-(y+.5)*cell-cell*.14,f'{index+1:02d}')
            for x in range(bw+1):
                canvas.setStrokeColorRGB(.18,.18,.18);canvas.setLineWidth(.7 if (x+x0)%5==0 else .25)
                canvas.line(left+x*cell,top,left+x*cell,top-bh*cell)
            for y in range(bh+1):
                canvas.setLineWidth(.7 if (y+y0)%5==0 else .25);canvas.line(left,top-y*cell,left+bw*cell,top-y*cell)
            canvas.setFillColorRGB(.08,.1,.14);canvas.setFont('Helvetica',min(7,cell*.42))
            for x in range(bw):canvas.drawCentredString(left+(x+.5)*cell,top+8,str(x0+x+1))
            for y in range(bh):canvas.drawRightString(left-8,top-(y+.5)*cell-cell*.14,str(y0+y+1))
            canvas.setFont('Helvetica',9);canvas.drawString(42,270,'IDs 01, 02 ... match B01, B02 ... in the inventory. No blank cells.')
            canvas.drawString(42,253,'Print at actual size. Do not enable fit-to-page. Check the 50 mm ruler below.')
            canvas.setStrokeColorRGB(0,0,0);canvas.setLineWidth(1);canvas.line(42,225,42+50*mm,225)
            canvas.line(42,221,42,229);canvas.line(42+50*mm,221,42+50*mm,229)
            canvas.drawString(42,207,'50 mm');canvas.drawRightString(550,30,f'Chart {page}/{pages}');canvas.showPage()
    canvas.save();return output.getvalue()

def _csv(rows):
    out=StringIO(newline='');writer=csv.writer(out)
    for row in rows:
        writer.writerow([("'"+v if isinstance(v,str) and v[:1] in ('=','+','-','@') else v) for v in row])
    return ('\ufeff'+out.getvalue()).encode('utf-8')

def export_files(pattern,cancel=None):
    rows=color_rows(pattern);counts=_csv([['id','code','name','hex','r','g','b','count']]+[[r['id'],r['code'],r['name'],r['hex'],*r['rgb'],r['count']] for r in rows])
    grid=_csv([['row/column',*range(1,pattern.labels.shape[1]+1)]]+[[y+1,*[f'B{int(v)+1:02d}' for v in row]] for y,row in enumerate(pattern.labels)])
    data={'version':1,'metrics':pattern.metrics,'pitch_mm':pattern.pitch_mm,'palette':rows,'grid':[[f'B{int(v)+1:02d}' for v in row] for row in pattern.labels]}
    buffer=BytesIO();Image.fromarray(preview_image(pattern)).save(buffer,format='PNG')
    files={'preview.png':buffer.getvalue(),'chart.png':chart_png(pattern,cancel),'chart.svg':chart_svg(pattern,cancel),
           'pattern.pdf':pattern_pdf(pattern,cancel),'colors.csv':counts,'grid.csv':grid,'pattern.json':json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'),
           'README.txt':('每格一粒豆，无空白格；图纸中的 01 对应清单 B01。行列从左上角 1 开始。\n'
                         'B01 等是项目内编号，自动配色 A01 等不是任何品牌的实体色号。实体色卡导入后请按 colors.csv 的 code 购豆。\n'
                         'PDF 按每页最多 29×29 格分页，按实际大小 100% 打印，用 50mm 标尺检查；格距不是熨烫后成品尺寸。\n'
                         '不做抖动；合并小色块会丢失细节。若五官消失，请降低最小色块，或提高网格和用色。\n'
                         '配色清单数量是净用量，采购时自行预留余量。照片及改编图的原授权要求仍适用。\n').encode('utf-8')}
    _cancel(cancel);return files

def package_files(files):
    output=BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,data in files.items():archive.writestr(name,data)
    return output.getvalue()

def palette_html(pattern):
    cards=[]
    for row in color_rows(pattern):
        cards.append(f'<div style="display:flex;gap:8px;align-items:center;padding:8px;border:1px solid #ddd;border-radius:8px"><span style="background:{row["hex"]};width:28px;height:28px;border:1px solid #aaa;flex-shrink:0"></span><span><b>{row["id"]}</b> · {html.escape(row["code"])}<br>{row["count"]} 粒 · {row["hex"]}</span></div>')
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px">'+''.join(cards)+'</div>'
