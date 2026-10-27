"""Validate all final artifacts and record measured, non-aesthetic evidence."""
from pathlib import Path
import json,hashlib,re,html
from urllib.parse import urlsplit,unquote
import xml.etree.ElementTree as ET
from PIL import Image,ImageOps
root=Path(__file__).resolve().parent;out=root/'outputs'
styles=['经典卡通','扁平插画','美式漫画','铅笔素描']
records=json.loads((root/'samples/metadata.json').read_text(encoding='utf-8'))['images']
rows=json.loads((out/'timings_v4.json').read_text(encoding='utf-8'))
assert len(records)==21 and len(rows)==84
images=[]
for record in records:
    source=root/'samples'/record['filename']
    with Image.open(source) as image:size=ImageOps.exif_transpose(image).size
    stem=source.stem
    for index,style in enumerate(styles,1):
        path=out/f'{index}_{style}'/(stem+'.png')
        with Image.open(path) as image:
            assert image.size==size and image.mode=='RGB' and image.format=='PNG';image.verify()
        images.append({'file':path.relative_to(root).as_posix(),'size':list(size),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with Image.open(out/(stem+'_comparison.jpg')) as image:image.verify()
for index in range(1,5):
    with Image.open(out/f'highres_{index}.png') as image:assert image.size==(3840,2880);image.verify()
review=out/'v4_review'
assert len(list(review.glob('*_crop.png')))==32
assert len(list(review.glob('*.jpg')))==32
details=json.loads((review/'detail_crop_coordinates.json').read_text(encoding='utf-8'))
assert len(details)==32
for detail in details:
    x,y,width,height=detail['crop_original_pixels'];ow,oh=detail['original_size']
    assert min(x,y)>=0 and min(width,height)>0 and x+width<=ow and y+height<=oh
broken=[]
for gallery in (out/'gallery.html',review/'gallery.html'):
    text=gallery.read_text(encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"]+)"',text):
        url=urlsplit(html.unescape(target))
        if url.scheme or not url.path:continue
        if not (gallery.parent/unquote(url.path)).resolve().exists():broken.append(target)
# The report is written below; this single expected forward link is intentional.
assert not [link for link in broken if not link.endswith('style_redesign_report.md')],broken
suite=ET.parse(out/'pytest_v4.xml').getroot().find('testsuite')
assert suite is not None and int(suite.attrib['tests'])==90 and int(suite.attrib['failures'])==0 and int(suite.attrib['errors'])==0
smoke=json.loads((out/'smoke_test.json').read_text(encoding='utf-8'))
assert smoke['http_status']==200 and len(smoke['exports'])==4 and smoke['full_png'] and smoke['batch_png_count']==2
assert '取消 True' in smoke['cancelled_batch_status']
bench=json.loads((out/'benchmark_v4.json').read_text(encoding='utf-8'));assert len(bench['rows'])==12
model=json.loads((root/'models/anime_metadata.json').read_text(encoding='utf-8'))
conversion=json.loads((out/'anime_conversion_verification.json').read_text(encoding='utf-8'))
runtime=json.loads((out/'runtime_cpu_v4.json').read_text(encoding='utf-8'))
assert runtime['providers']==['CPUExecutionProvider'] and runtime['pixel_max_difference_from_saved_result']==0
assert model['onnx_sha256']==conversion['onnx_sha256']
assert hashlib.sha256((root/'models/face_paint_v2.onnx').read_bytes()).hexdigest()==model['onnx_sha256']
errors={style:max(r['preview_export_mae'] for r in rows if r['style']==style) for style in styles}
delivery={'date':'2026-10-27','tests':90,'test_seconds':float(suite.attrib['time']),'sample_results':84,'comparison_sheets':21,'published_old_new_sheets':32,
    'native_detail_sheets':32,'highres_exports':4,'max_preview_export_mae_by_style':errors,'http_smoke':smoke,
    'model_sha256':model['onnx_sha256'],'reproduced_model_sha256':conversion['onnx_sha256'],'conversion_max_abs_error':conversion['max_abs_error'],
    'broken_gallery_links':[],'runtime_only_cpu_check':runtime,'images':images}
(out/'verification_v4.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# 四风格重做核验（2026-10-27）','','旧版硬量化卡通容易出现脏色块，素描依赖连续灰度明暗，用户反馈没有动漫感和线条感。本次替换核心处理路径，并根据真实照片放大检查进行了第二轮修正；没有用测试通过或低像素误差冒充艺术质量认可。','','## 实际改动','',
    '- 动漫：放弃第一轮仍像色块滤镜的传统赛璐璐试验，使用 Bryan Lee 的 FacePaint v2 预训练模型，由 ONNX Runtime CPUExecutionProvider 本地推理。无需训练，不上传照片，不依赖付费接口。复核 23 半身人像发现眉眼/笑容变形后，增加可靠 YuNet 定位的脸部上下文 512 局部细化；最多三张脸，跳过分析图中不足 50 像素的脸。只在生成后的绘画上补少量线条，不重新叠原照片的毛孔和皱纹。',
    '- 素描：白纸基底、结构张量方向场、法线 DoG/切线平滑、主轮廓与局部线稿；沿结构方向的确定性噪声积分形成排线。取消旧版大面积灰度照片底色和规则全局斜线。复核发现平坦深背景有伪笔触后，去掉平坦区域排线；修复纯黑区域的除零伪墨块。',
    '- 插画：加大软权重 Kuwahara 的区域尺度，先合并小岛再保留五官标签；避免小半径平滑仍带入大量皮肤碎块。仍是有意使用有限调色板的插画，与动漫模型不同。',
    '- 漫画：区域平滑、独立饱和度/明度分级、顺结构粗线、较浓阴影和中间调半调网点；不调用动漫模型。',
    '- 高清：所有档位复用统一绘画结构，按原尺寸分块导出，去掉把原照片细纹混回绘画的做法。动漫的精细档只是双三次插值，界面明确标注。',
    '- 保持旧 API 内部风格键/目录名；界面显示动漫人像与线条素描；拼豆算法、预设、图纸格式没有改动。','','## 实际执行记录','',
    f'- 最终完整 pytest：**90 项通过**，{float(suite.attrib["time"]):.2f} 秒；报告 outputs/pytest_v4.xml。此前二次视觉修正的专项检查另有 29 项通过，11 项未选中，未把未选中计为通过。',
    '- 全部 21 张正式授权照片 × 4 风格，84 张原尺寸 PNG；21 张原图加四结果的对比；8 张代表照片 × 4 风格的 32 张新旧对比和 32 张五官/结构局部；另有 4 张真实 3840×2880 导出。脸部细化改动后单独重算全部 21 张动漫结果，另外 63 张未改算法结果保留此前真实记录，未伪造重算耗时。',
    '- 旧版对比直接读取固定提交 0b46ac10b7dd8174afe8502b769ee8f9bba409dd 中已经发布的 PNG，避免在新依赖下重跑旧算法造成基线漂移。无需本地备份。',
    '- 实际启动 Gradio，经 HTTP 检查 4 种风格的两阶段预览、原尺寸 JPG/PNG（含 sRGB）、双图 ZIP、进行中的批任务取消；应用正常关闭。原始记录 outputs/smoke_test.json。',
    '- 运行依赖检查 pip check 通过。完成转换后已卸载仅开发使用的 PyTorch、ONNX、networkx、ml_dtypes；在不含这些开发包的环境中实际重算 23 半身人像，与保存结果逐像素完全一致，且只启用 CPUExecutionProvider（outputs/runtime_cpu_v4.json）。运行 requirements/lock 不包含上述转换依赖。',
    f'- 从固定源码与 SHA256 权重再次运行交付的转换工具，复现 ONNX 校验和一致；CPU 原模型与 ONNX 在 192×256、64×100、256×192 输入上比较，最大绝对误差 {conversion["max_abs_error"]:.8f}（归一化输出，阈值 0.001）。',
    '- 输出尺寸、格式、84 个 PNG 校验和、画廊链接均核验；人脸框由 1000 分析坐标换算到原始像素后裁剪，32 张局部图的坐标记录在 outputs/v4_review/detail_crop_coordinates.json；证据 outputs/verification_v4.json。','','## 视觉核验与仍存在的问题','',
    '代表样片覆盖 01 老人近裁/硬光、03 三分之二侧脸、05 儿童/较深肤色、13 室外多人复杂背景、15 山景、17 建筑、19 动物、23 官方人像。请打开 outputs/v4_review/gallery.html 自行放大看原图 / 旧版 / 新版，所有样片均在同页，不只展示最好的一张。',
    '', '观察到的变化：单人人像的眼睛和头发呈现绘画结构，旧卡通的全脸硬色块被替换；素描的眼睛、鼻翼、唇线和衣褶明显可见，并保留纸白；增大插画平滑区域后，细皱纹形成的小岛减少；漫画的粗线、阴影和网点更明确。',
    '', '局限也需直说：动漫模型会改变眉眼、脸形、虹膜表现和年龄感，不能保证身份完全一致，尤其多人小脸可能变形；非人像可能偏暗或过度抽象，17 建筑样片的招牌文字和钟面细节也有损失，不适合保真记录。深皱纹的素描/漫画仍会显得重，复杂背景和发丝会保留程序化细纹，可降低内部细线或增加平滑；这不是专业人工素描。插画仍会损失渐变，个别肤色区域有分级，不能宣称彻底消除所有色块。',
    '', '动漫全图推理最长边 640，可靠的中小脸另用 512 区域细化，传统分析最长边 1000。4K 文件保持原尺寸，但不是在 4K 上逐像素运行神经网络，也不是超分辨率；画稿之外的真实细节不会凭空出现。只适合在这个明确 CPU 成本和质量取舍下使用。没有训练新模型，也没有虚构人工盲评分数。','','## CPU 实测与预览一致性','',
    '独立子进程对 640×480、1920×1080、3840×2160 测量四算法，冷启动、预热后三次中位数、缓存预览及 5ms 采样 RSS 峰值见 benchmark_v4.md 和 outputs/benchmark_v4.json。全量评估的单图时间不作为独立性能基准。',
    '', '| 风格（内部键） | 全样片预览 / 缩小导出最大 MAE |','|---|---:|']
for style,value in errors.items():lines.append(f'| {style} | {value:.3f} |')
lines+=['','误差基于 0～255 RGB 的平均绝对差，只检查尺度/结构一致性，不衡量审美；不同插值和输出尺度网点会产生差异。','','## 来源与复现','',
    '- 模型作者仓库：https://github.com/bryandlee/animegan2-pytorch ，固定提交 25d7b017267208dfaf34026aa3425e518372aa2f；采用其中作者发布的 FacePaint v2，不是原 AnimeGAN 的风景转换权重。附 MIT 许可与原权重/ONNX 校验和，详见 models/anime_metadata.json、models/LICENSE-FacePaint.txt。',
    '- 结构引导滤波参考原论文：https://www.kyprianidis.com/p/tpcg2008/jkyprian-tpcg2008.pdf 。NPR 代码为本项目 NumPy/OpenCV 实现。',
    '- 照片署名和许可：samples/metadata.json、samples/ATTRIBUTION.md；模型不会改变照片再分发应履行的署名/相同方式共享要求。',
    '- 复现命令、安装、启动及高清限制见 README.md。历史 v2/v3 报告保留，当前版本请看 v4。']
(root/'style_redesign_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in delivery.items() if k not in ('images','http_smoke')},ensure_ascii=False,indent=2))
