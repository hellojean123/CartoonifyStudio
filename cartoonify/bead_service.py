"""Bead exports share the existing CPU gate, cancellation and decode protections."""
from pathlib import Path
import time
import uuid
from .beads import BeadParams,make_pattern,read_palette,export_files,package_files,preview_image,palette_html
from .service import EXPORTS,job,load_cached

def create_export(path,grid,colors,smooth,min_region,cohesion,pitch,preserve,mode,custom_path,session):
    if not path:raise ValueError('请先上传照片，或从内置照片选择原图')
    if mode not in ('自动精简颜色','使用实体色卡 CSV'):raise ValueError('未知色卡模式')
    if mode=='使用实体色卡 CSV' and not custom_path:raise ValueError('请上传实体色卡 CSV，包含 code,r,g,b 列')
    started=time.perf_counter();created=[]
    with job(session) as token:
        palette=read_palette(custom_path) if mode=='使用实体色卡 CSV' else None
        image=load_cached(path);p=BeadParams(grid,colors,smooth,min_region,cohesion,pitch,preserve)
        pattern=make_pattern(image,p,palette,cancel=token.is_set)
        files=export_files(pattern,cancel=token.is_set);bundle=package_files(files)
        if token.is_set():raise InterruptedError('拼豆生成已取消')
        EXPORTS.mkdir(parents=True,exist_ok=True);stem='beads-'+uuid.uuid4().hex
        try:
            for suffix,data in (('.pdf',files['pattern.pdf']),('.zip',bundle)):
                dest=EXPORTS/(stem+suffix);tmp=dest.with_suffix(dest.suffix+'.part');created.append(tmp)
                tmp.write_bytes(data)
                if token.is_set():raise InterruptedError('拼豆生成已取消')
                tmp.replace(dest);created.append(dest)
        except BaseException:
            for item in created:item.unlink(missing_ok=True)
            raise
    m=pattern.metrics;b=m['before'];a=m['after']
    status=(f"{time.perf_counter()-started:.2f} 秒 · {m['width']}×{m['height']} 格 · {m['used_colors']} 色 · {m['total_beads']} 粒\n"
            f"孤立单粒：{b['isolated_beads']} → {a['isolated_beads']}；小色块：{b['small_regions']} → {a['small_regions']}。"
            f"保留五官细节格 {m['protected_feature_cells']}；合并仍可能简化发丝等细节。")
    return preview_image(pattern),palette_html(pattern),str(EXPORTS/(stem+'.pdf')),str(EXPORTS/(stem+'.zip')),status
