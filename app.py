"""Cartoonify Studio v4: staged previews and one shared CPU resource queue."""
import time
from dataclasses import replace
import uuid
import os
import json
from pathlib import Path
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
os.environ.setdefault('GRADIO_TEMP_DIR',str(Path(__file__).resolve().parent/'outputs'/'ui_temp'))
import gradio as gr
from cartoonify.core import STYLES,PRESETS,Params,process,resize,CACHE
from cartoonify.beads import PRESETS as BEAD_PRESETS
from cartoonify.bead_service import create_export as bead_export
from cartoonify.service import ROOT,load_cached,job,cancel_job,export_one,batch_export

def make_params(outline,smooth,colors,contrast,colored,protect,face_strength,outer,inner,spacing,angle,opacity,shade,paper,boxes,anime_mode='动漫增强'):
    return Params(outline,smooth,colors,contrast,colored,protect,face_strength,outer,inner,spacing,angle,opacity,shade,paper,boxes,anime_mode)

def preview(path,style,*values):
    if not path: yield None,None,None,'请上传或选择测试照片。';return
    session=values[-1];p=make_params(*values[:-1]);start=time.perf_counter()
    try:
        with job(session) as token:
            image=load_cached(path)
            rough_params=replace(p,protect=False) if style==STYLES[0] else p
            rough=process(resize(image,480),style,rough_params,preview=True,quality='标准',cancel=token.is_set)
            yield resize(image,1000),rough,None,f'{time.perf_counter()-start:.2f} 秒 · 快速草稿，正在细化（草稿色板可能不同）'
            if token.is_set(): yield gr.skip(),gr.skip(),None,'已取消';return
            info={}
            result=process(image,style,p,preview=True,quality='标准',cancel=token.is_set,diagnostics=info)
            yield resize(image,1000),result,None,f'{time.perf_counter()-start:.2f} 秒 · 最终预览 {result.shape[1]}×{result.shape[0]} · 保护区域 {len(info["faces"])} · 动漫脸部细化 {info.get("anime_refined_faces",0)} · 缓存命中 {info["cache"].get("hits",0)}'
    except InterruptedError: yield gr.skip(),gr.skip(),None,'处理已取消'
    except Exception as exc: raise gr.Error(str(exc)) from exc

def export(path,style,*values):
    if not path: raise gr.Error('请先上传照片')
    parameter_values=values[:-3];fmt,quality,session=values[-3:]
    p=make_params(*parameter_values);start=time.perf_counter()
    try:
        with job(session) as token:
            result,dest=export_one(path,style,p,fmt,quality,token.is_set)
        return resize(load_cached(path),1000),resize(result,1000),str(dest),f'{time.perf_counter()-start:.2f} 秒 · {quality}原尺寸导出 {result.shape[1]}×{result.shape[0]}'
    except Exception as exc: raise gr.Error(str(exc)) from exc

def batch(paths,style,*values,progress=gr.Progress()):
    p=make_params(*values[:-3]);fmt,quality,session=values[-3:]
    try: return batch_export(paths,style,p,fmt,quality,session,progress)
    except Exception as exc: raise gr.Error(str(exc)) from exc

def defaults(style):
    index=STYLES.index(style);a,b,c,d=PRESETS[index]
    return (a,b,gr.update(value=c,minimum=4,maximum=12 if index==2 else 48,visible=index in (1,2),label='明度分级' if index==2 else '调色板颜色数'),d,
            gr.update(value=False,visible=index==3),True,.75,.8,.5,7,15,.28,.48,.08,'',gr.update(value='动漫增强',visible=index==0),gr.update(visible=index==2),gr.update(visible=index==3))

def manual_box(left,top,width,height):
    if left+width>100 or top+height>100: raise gr.Error('保护框超出图片边界，请缩小宽度或高度')
    return json.dumps([[left/100,top/100,width/100,height/100]])

def bead_generate(path,grid,colors,smooth,min_region,cohesion,pitch,preserve,mode,custom,session):
    try:return bead_export(path,grid,colors,smooth,min_region,cohesion,pitch,preserve,mode,custom,session)
    except InterruptedError:return gr.skip(),gr.skip(),None,None,'拼豆生成已取消'
    except Exception as exc:raise gr.Error(str(exc)) from exc


def build_app():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue='violet'),title='Cartoonify Studio',css='.gradio-container{max-width:1480px!important} footer{display:none!important}') as demo:
        session=gr.State(lambda:uuid.uuid4().hex)
        gr.Markdown('# 🎨 Cartoonify Studio\n纯 CPU · 动漫人像 · 铅笔素描 · 高清导出 · 拼豆图纸')
        source=gr.File(label='上传原照片 · JPG / PNG / WebP',file_types=['.jpg','.jpeg','.png','.webp'],type='filepath')
        with gr.Tab('照片卡通化'):
            with gr.Row():
                with gr.Column(scale=1):
                    style=gr.Dropdown([('动漫人像（双模型）',STYLES[0]),('扁平插画',STYLES[1]),('美式漫画',STYLES[2]),('铅笔素描（变宽笔触）',STYLES[3])],value=STYLES[0],label='风格与预设')
                    anime_mode=gr.Radio(['动漫增强','柔和插画'],value='动漫增强',label='动漫表现',info='动漫增强使用独立的 Webtoon 预训练模型；可能改变脸形、年龄感和肤色。柔和插画保留原版模型。')
                    outline=gr.Slider(0,1,.35,step=.05,label='整体轮廓强度')
                    smooth=gr.Slider(1,10,5,step=1,label='区域平滑')
                    colors=gr.Slider(4,48,16,step=1,label='调色板颜色数',visible=False)
                    contrast=gr.Slider(.6,1.8,1.,step=.01,label='对比度')
                    colored=gr.Checkbox(label='彩色铅笔',visible=False)
                    protect=gr.Checkbox(value=True,label='保护人像（自动检测 + 手动区域）')
                    face_strength=gr.Slider(0,1,.75,step=.05,label='人像保护力度')
                    with gr.Accordion('轮廓分层与手动保护',open=False):
                        outer=gr.Slider(0,1,.8,step=.05,label='主要边缘 / 粗线')
                        inner=gr.Slider(0,1,.5,step=.05,label='内部纹理 / 细线')
                        gr.Markdown('自动检测漏掉侧脸时，可按图片百分比添加一个保护框。')
                        with gr.Row():
                            left=gr.Slider(0,95,25,step=1,label='左侧位置 %')
                            top=gr.Slider(0,95,15,step=1,label='顶部位置 %')
                        with gr.Row():
                            box_width=gr.Slider(1,100,40,step=1,label='保护宽度 %')
                            box_height=gr.Slider(1,100,55,step=1,label='保护高度 %')
                        add_box=gr.Button('应用手动保护框')
                        clear_box=gr.Button('清除手动保护框')
                        boxes=gr.Textbox(value='',visible=False)
                    with gr.Group(visible=False) as comic_group:
                        spacing=gr.Slider(4,18,7,step=.5,label='网点间距（1000 边长参考像素）')
                        angle=gr.Slider(0,90,15,step=1,label='网点角度')
                        opacity=gr.Slider(0,.65,.28,step=.01,label='网点浓度')
                    with gr.Group(visible=False) as pencil_group:
                        shade=gr.Slider(0,.8,.48,step=.02,label='铅笔明暗与排线浓度',info='设为 0 只保留结构线；细笔触连续变宽变淡，暗部以轻笔叠加。')
                        paper=gr.Slider(0,.4,.08,step=.01,label='纸张颗粒')
                    reset=gr.Button('恢复当前风格默认参数')
                    preview_btn=gr.Button('重新预览',variant='primary')
                    quality=gr.Radio(['标准','精细'],value='精细',label='动漫高清插值',info='两档共用同一张动漫结构图；精细使用双三次插值。不会恢复原照片纹理或凭空增加细节。')
                    fmt=gr.Radio(['PNG','JPG'],value='PNG',label='导出格式')
                    export_btn=gr.Button('处理原始尺寸并导出')
                    stop=gr.Button('取消当前任务')
                with gr.Column(scale=3):
                    with gr.Row():
                        before=gr.Image(label='原图预览',interactive=False)
                        after=gr.Image(label='效果预览',interactive=False)
                    status=gr.Textbox(label='状态与耗时',interactive=False)
                    download=gr.File(label='下载原尺寸结果')
                    gr.Markdown('松开滑块后自动预览：先草稿，再更新统一色板的最终预览。图片和动漫模型均在本机 CPU 处理，不上传照片。动漫可选更强的 Webtoon 或柔和插画模型，均为本地预训练模型；人物特征可能变化；输出尺寸不等于新增真实细节。')
        params=[outline,smooth,colors,contrast,colored,protect,face_strength,outer,inner,spacing,angle,opacity,shade,paper,boxes,anime_mode]
        outputs=[before,after,download,status]
        triggers=[preview_btn.click,source.change]+[s.release for s in [outline,smooth,colors,contrast,face_strength,outer,inner,spacing,angle,opacity,shade,paper]]+[colored.change,protect.change,boxes.submit,anime_mode.change]
        preview_event=gr.on(triggers=triggers,fn=preview,inputs=[source,style,*params,session],outputs=outputs,trigger_mode='always_last',concurrency_id='cpu',concurrency_limit=1,api_name='preview')
        style.change(lambda selected:gr.update(visible=selected==STYLES[0]),style,quality,queue=False,api_name=False)
        style_event=style.change(defaults,style,[*params,comic_group,pencil_group],queue=False,api_name='style_defaults')
        style_event.then(preview,[source,style,*params,session],outputs,concurrency_id='cpu',concurrency_limit=1,trigger_mode='always_last',api_name=False)
        reset.click(defaults,style,[*params,comic_group,pencil_group],queue=False,api_name=False)
        add_box.click(manual_box,[left,top,box_width,box_height],boxes,queue=False,api_name=False).then(preview,[source,style,*params,session],outputs,concurrency_id='cpu',concurrency_limit=1,api_name=False)
        clear_box.click(lambda:'',None,boxes,queue=False,api_name=False).then(preview,[source,style,*params,session],outputs,concurrency_id='cpu',concurrency_limit=1,api_name=False)
        export_event=export_btn.click(export,[source,style,*params,fmt,quality,session],outputs,concurrency_id='cpu',concurrency_limit=1,api_name='export')
        with gr.Tab('拼豆图纸'):
            bead_session=gr.State(lambda:uuid.uuid4().hex)
            gr.Markdown('## 把照片变成更好拼的图纸\n使用上方原图；每格一粒豆。优先减少孤立杂色与碎块，默认不做抖动。')
            with gr.Row():
                with gr.Column(scale=1):
                    bead_preset=gr.Dropdown(list(BEAD_PRESETS),value='人像清晰（推荐）',label='拼装难度预设')
                    bead_grid=gr.Slider(16,160,96,step=1,label='最长边格数',info='保持原图比例；格数越多，细节与工作量越大。')
                    bead_colors=gr.Slider(2,32,18,step=1,label='最多使用几种颜色')
                    bead_smooth=gr.Slider(0,4,1,step=1,label='先平滑照片',info='减少毛发、纹理带来的零碎颜色。')
                    bead_min=gr.Slider(0,12,3,step=1,label='合并小于几粒的色块',info='按上下左右相连计算；调高更好拼，但可能合并眼睛等细节。')
                    bead_cohesion=gr.Slider(0,4,1.5,step=.5,label='相邻颜色聚合力度')
                    bead_pitch=gr.Radio([5.,2.6],value=5.,label='PDF 每格间距（毫米）',info='按实际大小 100% 打印，每页最多 29×29 格；格距不等于熨烫后尺寸。')
                    bead_preserve=gr.Checkbox(value=True,label='保留五官、肤色与面部明暗',info='人脸采用更轻的平滑与合并，并优先分配肤色；会增加少量必要小色块。')
                    bead_mode=gr.Radio(['自动精简颜色','使用实体色卡 CSV'],value='自动精简颜色',label='配色方式')
                    bead_custom=gr.File(label='实体色卡 CSV：code,r,g,b，可加 name',file_types=['.csv'],type='filepath',visible=False)
                    gr.Markdown('自动颜色只用于设计参考，不对应品牌色号。导入实体色卡后，程序只从该色卡选色；B01 等仍是图纸内编号。')
                    bead_go=gr.Button('生成 / 更新拼豆图纸',variant='primary')
                    bead_stop=gr.Button('取消拼豆生成')
                with gr.Column(scale=3):
                    bead_preview=gr.Image(label='拼豆配色预览（放大像素，不增加新颜色）',interactive=False)
                    bead_status=gr.Textbox(label='格数、用量与碎色块变化',interactive=False)
                    bead_legend=gr.HTML(label='颜色与净用量')
                    with gr.Row():
                        bead_pdf=gr.File(label='可打印 A4 图纸 PDF')
                        bead_zip=gr.File(label='完整图纸包：PDF / PNG / SVG / 色号用量 / 网格 CSV')
            bead_preset.change(lambda name:BEAD_PRESETS[name],bead_preset,[bead_grid,bead_colors,bead_smooth,bead_min,bead_cohesion],queue=False,api_name=False)
            bead_mode.change(lambda mode:gr.update(visible=mode=='使用实体色卡 CSV'),bead_mode,bead_custom,queue=False,api_name=False)
            bead_go.click(bead_generate,[source,bead_grid,bead_colors,bead_smooth,bead_min,bead_cohesion,bead_pitch,bead_preserve,bead_mode,bead_custom,bead_session],
                          [bead_preview,bead_legend,bead_pdf,bead_zip,bead_status],concurrency_id='cpu',concurrency_limit=1,api_name='beads')
            bead_stop.click(cancel_job,bead_session,bead_status,queue=False,api_name='beads_cancel')
        with gr.Tab('批量处理'):
            files=gr.File(file_count='multiple',file_types=['.jpg','.jpeg','.png','.webp'],label='多选照片')
            go=gr.Button('批量生成原尺寸 ZIP')
            zip_out=gr.File(label='下载 ZIP');batch_status=gr.Textbox(label='批量状态')
            batch_event=go.click(batch,[files,style,*params,fmt,quality,session],[zip_out,batch_status],concurrency_id='cpu',concurrency_limit=1,api_name='batch')
        stop.click(cancel_job,session,status,queue=False,api_name='cancel')
        with gr.Tab('内置测试照片'):
            samples=sorted((ROOT/'samples').glob('*.jpg'))
            gr.Examples([[str(p)] for p in samples],[source],label='点击文件载入')
            gr.Gallery(value=[str(p) for p in samples],columns=5,label='授权照片画廊')
        gr.Markdown('输入上限 1600 万像素 / 40 MB；透明区域白底；方向与色彩自动处理。自动人像检测可能漏检侧脸，可添加手动保护框。缓存有内存上限；精细档分块重建，支持取消。')
    return demo

if __name__=='__main__': build_app().queue(default_concurrency_limit=1,max_size=8).launch(server_name='127.0.0.1',max_file_size='40mb',inbrowser=True)
