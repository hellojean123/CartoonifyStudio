# Cartoonify Studio v3

Windows 11 / Python 3.12 / 纯 CPU 照片卡通化。OpenCV、NumPy、Pillow、Gradio、pytest；无 CUDA、无训练、无付费 API。算法与界面独立，处理不上传照片，界面统计默认关闭。

![原图与四种风格对比](outputs/03_portrait_comparison.jpg)

## 安装和启动（Windows PowerShell）

本机项目位于 `D:\CartoonifyStudio`，有独立 Python 3.12.10 `runtime` 和 `.venv`。双击 `start.bat`，或运行：

```powershell
Set-Location D:\CartoonifyStudio
.\.venv\Scripts\python.exe app.py
```

全新机器先安装官方 Python 3.12，再运行：

```powershell
Set-Location D:\CartoonifyStudio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

无需激活环境或修改 PowerShell 执行策略。浏览器默认打开本机地址 http://127.0.0.1:7860 ，端口占用时以控制台显示为准。关闭控制台停止服务。`requirements-lock.txt` 记录本机实际依赖版本，迁移后须重建虚拟环境。

## 操作流程

1. 拖入 JPG / PNG / WebP，或在内置测试照片页选择文件。
2. 选择经典卡通、扁平插画、美式漫画或铅笔素描，切换风格载入合理预设。
3. 松开滑块自动预览，先显示 480 边长草稿，再显示最长边 1000 的最终预览。草稿是临时近似，色板可能不同；最终预览和导出使用同一张分析图确定色板与主体轮廓。
4. 可分别调整整体轮廓、主要粗线、内部细线。人像保护默认打开，减少皮肤细纹和五官过黑。
5. 自动检测漏掉侧脸时，展开轮廓与手动保护选项，按图片百分比设置左、上、宽、高，点击应用；手动框与自动检测区域叠加。可清除手动框。保护框是羽化区域，并非精确皮肤分割。
6. 插画显示“调色板颜色数”4～48；漫画显示“明度分级”4～12，并提供网点间距、角度、浓度；素描提供彩铅、石墨明暗和纸张颗粒。无效的风格参数隐藏。
7. 经典卡通可选择标准或精细细节恢复（其他风格隐藏此选项）；选择 PNG 或 JPG，点击原始尺寸导出，再下载文件。预览不会不断写入导出文件。
8. 批量页使用当前参数逐张处理并打包 ZIP，显示进度；失败图片保留原因，已成功图片仍可下载。点击取消，在图块或图片边界停止，保留有效的部分 ZIP。

“恢复当前风格默认参数”恢复风格值并清除手动框；下一次预览采用恢复值。自动预览仅在松开滑块时触发，同一事件只保留最新待处理请求。取消针对当前正在处理的任务，不等同关闭程序或删除已导出文件。

## 高清、尺度与内存

- 两种导出档位均保持输入原始尺寸，不再将 1400 的整幅完成图简单放大作为精细结果。
- 标准：复用统一分析层，在最终尺寸重绘轮廓、抗锯齿网点、石墨笔触与纸纹。
- 精细：经典卡通仅用 20% 原图引导恢复，控制照片细纹回流；插画、漫画、素描与标准使用相同结构重建，保留色块/分级/明暗，不再混回连续照片纹理。两档都保持原尺寸，按带重叠边界的 512 图块渲染。
- 网点间距以最长边 1000 为参考，随输出尺寸缩放；预览与导出色板一致。经典卡通精细模式少量增加细节，所以与标准预览不要求逐像素相同。
- 分析图最长边 1000，属于传统多尺度抽象；不能恢复原图中不存在的细节，也不等同生成式重绘。
- 输入上限 1600 万像素、40 MB、单边 20000；超过时明确拒绝。透明度白底、EXIF 方向纠正、ICC 转为 sRGB，导出附 sRGB 配置。
- 解码缓存上限 64 MiB，中间层缓存上限 112 MiB；缓存按内容和影响阶段的参数区分，只调轮廓/网点可复用区域与色板。缓存跨任务复用，退出程序释放。
- 预览、导出和批处理共享 CPU 队列，另有应用资源锁；OpenCV 线程 2，OpenCL 关闭。输入和最终输出仍需驻留内存，缓存上限不等于进程总内存上限。

## 四种处理路线

| 风格 | 实际流程 |
|---|---|
| 经典卡通 | 双边滤波、自适应阈值、颜色层次、粗细轮廓融合 |
| 扁平插画 | Mean Shift 区域平滑、Lab 采样 K-Means、32³ 色彩查找表、局部空间投票、相邻孤立色块合并、轻微线条；不混回连续照片渐变 |
| 美式漫画 | 明度分级、独立粗细轮廓、大面积阴影与中间调网点、可调角度/间距/浓度、边缘抗锯齿 |
| 铅笔素描 | XDoG 风格软阈值线条、主要轮廓、连续石墨明暗、局部梯度调制的不规则细笔触、确定性纸张颗粒、可选彩铅 |

人像检测优先使用随项目分发的 YuNet 2023 预训练小模型（约 227 KiB，MIT），指定 OpenCV CPU 后端；无需训练、无需联网，不含人脸识别。传统 Haar 级联补充检测，模型缺失时自动回退。模型来源、上游提交和 SHA256 见 models/metadata.json，许可见 models/LICENSE-YuNet.txt。检测五点用于眼鼻嘴保护，皮肤使用框内宽松色彩约束；并非精确语义分割，对小脸、遮挡和强逆光仍可能失败。主要/内部轮廓按不同空间尺度提取，并不是语义意义上的精确外轮廓。

## 测试和评估命令

```powershell
Set-Location D:\CartoonifyStudio
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/pytest.xml
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe evaluate_v3.py
.\.venv\Scripts\python.exe benchmark_v3.py
.\.venv\Scripts\python.exe validate_delivery.py
Start-Process .\outputs\gallery.html
Start-Process .\outputs\blind_review.html
```

`evaluate_v3.py` 处理 21 张正式样本的四种风格，生成 84 张 PNG、21 张五联对比、28 张 v2/v3 新旧对照及 28 张原尺寸局部图（outputs/v3_review/gallery.html），并对真实高清参考生成 4 张原尺寸效果。需要 Git 和保留的初始提交读取 v2 基线；缺失时明确失败。`blind_review.html` 可选择 A/B/相当，评分仅保存在浏览器，可导出 JSON；映射答案另存 `blind_key.json`，程序不伪造用户评分。

`benchmark_v3.py` 对 640×480、1920×1080、3840×2160 四风格使用与历史 v2 相同真实高分辨率输入，独立进程预热后测量 3 次、记录中位数/P95 和 RSS；另测缓存调参和量化。只有 3 次的 P95 是描述性数字，不是长期服务尾延迟。

`outputs/timings_v3.json` 有全部照片处理分阶段耗时、人像框和预览/导出差异；timings.json 与 benchmark.md 保留为历史 v2 数据，当前性能见 benchmark_v3.md。指标是诊断工具，低碎线不自动等于艺术效果更好；人工盲评仍有必要。

## 照片与许可

21 张正式样本（16 张人物）及完整来源/作者/许可/下载日期见 `samples/metadata.json`、`samples/ATTRIBUTION.md`。真实 3840×2880 评估图单独位于 `samples/high_resolution`，包括原始分辨率、官方缩略图地址和署名；它来自更高分辨率原图，而非低清图放大。

接受 CC0、公有领域、CC BY、CC BY-SA、英国 OGL 3；排除 NC/ND 和无法核实作者的图片。再分发时保留来源和署名，CC BY-SA 改编结果遵守对应版本相同许可。人物仅供技术演示，不表示背书。初始失败和拒绝候选均有记录，候选存档不计入正式评估。

```powershell
.\.venv\Scripts\python.exe download_samples.py
.\.venv\Scripts\python.exe fetch_highres.py
```

已有完整元数据时，下载脚本刷新已核实的样本；首次搜索仍需内容复核，不能仅凭标题认定是真实人物摄影。

## 文件与评估基线

- `cartoonify/core.py`：算法、保护区域、分层与分块渲染。
- `cartoonify/cache.py`：按字节限制的只读缓存。
- `cartoonify/service.py`：资源锁、取消、解码和导出，独立于 Gradio。
- `app.py`：WebUI；`tests/`：回归与故障测试。
- `outputs/gallery.html`：全量结果；`outputs/upgrade_comparisons`：新旧对照与原生高清裁剪。
- `benchmark.md`、`upgrade_report.md`、`test_report.md`：真实性能、七项升级与交付验证。
- `benchmarks/legacy_v1.py`：供新旧对照的最小旧版算法参考，不依赖整套本地备份。

导出文件在 `outputs/exports`，可自行清理历史生成文件。迁移或从 GitHub 克隆后重建 `.venv`；仓库包含算法参考、正式样本、授权记录及效果，不包含虚拟环境、Python 安装包、缓存、拒绝候选和整套本地备份。不要覆盖其他项目目录。

## 已知限制

脸部保护是粗区域，不是语义分割；非常复杂背景仍可能出现线条；插画仍会牺牲部分色彩渐变；网点缩小时可能有摩尔纹；排线是程序化纹理，不能等同手绘笔触。精细高清比旧版单纯放大承担更多工作，可能更慢。具体时间、内存和预览误差以本机报告为准。

## v3 核验与限制

本轮以人像结构、素描明暗、独立风格和高清一致性为先，改动及实际检查见 optimization_v3.md。新版改善不等于每张照片都胜过旧版：深皱纹、强硬光仍可能形成较重色块；扁平插画会有意舍弃发丝细节；素描仍属传统滤镜，笔触不等同人工绘制。网页盲评没有预置或虚构的偏好分数。

## 照片转拼豆图纸

上方上传原照片，再切换“拼豆图纸”页，点击“生成 / 更新拼豆图纸”。算法独立于卡通四风格，直接从原照片生成网格，避免卡通碎线再次被量化成杂色。

- 默认“人像清晰”：96 格 / 18 色 / 轻度平滑 / 3 粒，保留五官和面部明暗。另有轻松拼 72 格 / 10 色 / 5 粒、均衡 96 格 / 16 色 / 3 粒、保留细节 128 格 / 24 色 / 1 粒。人像预设增加格数和颜色，会增加用量。
- 网格保持照片比例，最长边可调 16～160 格；每格一粒豆，所有格子都计入用量，白色也不是空白。
- 使用浮点 CIELab 量化、无抖动、相邻颜色聚合与四连通小区域合并，减少孤立杂色。调整“相邻颜色聚合力度”和“合并小于几粒的色块”可控制简洁程度。
- 默认保留眼睛、鼻子、嘴唇的亮暗结构；人脸轻度平滑、优先分配肤色，并降低区域合并强度。这样可能保留少量单粒；关闭该选项可进一步减少碎块。检测失败、非常小的脸仍可能损失五官；可提高格数、用色，降低小块阈值，或先裁剪主体再上传。
- 下载包包含 preview.png、编号 chart.png、矢量 chart.svg、A4 分页 pattern.pdf、colors.csv 用量清单、grid.csv 网格和 pattern.json。同一份网格驱动所有文件，编号 01 对应清单 B01，行列从左上角 1 开始。
- PDF 每页最多 29×29 格，支持 5mm / 2.6mm 格距；按“实际大小 / 100%”打印，用图内 50mm 标尺检查，不启用适应纸张。格距不是熨烫后的成品尺寸。2.6mm 小格的编号较小，可放大阅读。
- 颜色数量是净用量，购买时自行预留余量。自动调色 A01 等只是设计参考，B01 等是本图内编号，都不代表任何品牌的实体豆色号。

要使用自己的实体豆，选择“使用实体色卡 CSV”，上传 UTF-8 CSV，列为 `code,r,g,b`，可增加 `name`。最多 256 条色卡颜色，最终使用不超过滑块指定数量，只使用导入色卡的 RGB；不会编造品牌色号。示例 `palettes/example.csv` 全部是 EXAMPLE 颜色，不是实际品牌色卡。屏幕颜色与实物受光照影响，仍应核对实体色卡。含非 ASCII 色号的完整信息保存在 CSV / JSON 中；PDF 用本地编号对应，便于跨系统打印。

```powershell
Set-Location D:\CartoonifyStudio
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe evaluate_beads.py
.\.venv\Scripts\python.exe benchmark_beads.py
.\.venv\Scripts\python.exe smoke_beads.py
Start-Process .\outputs\beads\gallery.html
```

21 张真实样片均已生成拼豆图纸与对比，见 `outputs/beads/gallery.html`；算法与实测报告见 `beads_report.md`。原照片作者、许可在画廊及示例包 SOURCE.txt 中保留，改编结果继续遵守原照片授权。

面部细节专项：`bead_face_report.md`；原图 / 旧版 / 同格数同颜色新算法 / 新人像预设对照见 `outputs/beads_face_v2/gallery.html`。低层 `BeadParams()` 保持旧的 72 格/10 色默认，UI 和正式样例使用人像清晰预设。
