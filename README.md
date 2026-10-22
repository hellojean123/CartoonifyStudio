# Cartoonify Studio v2

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
6. 插画显示“调色板颜色数”4～48；漫画显示“明度分级”4～12，并提供网点间距、角度、浓度；素描提供彩铅、阴影排线和纸张颗粒。无效的风格参数隐藏。
7. 选择标准或精细质量、PNG 或 JPG，点击原始尺寸导出，再下载文件。预览不会不断写入导出文件。
8. 批量页使用当前参数逐张处理并打包 ZIP，显示进度；失败图片保留原因，已成功图片仍可下载。点击取消，在图块或图片边界停止，保留有效的部分 ZIP。

“恢复当前风格默认参数”恢复风格值并清除手动框；下一次预览采用恢复值。自动预览仅在松开滑块时触发，同一事件只保留最新待处理请求。取消针对当前正在处理的任务，不等同关闭程序或删除已导出文件。

## 高清、尺度与内存

- 两种导出档位均保持输入原始尺寸，不再将 1400 的整幅完成图简单放大作为精细结果。
- 标准：复用统一分析层，在最终尺寸重绘轮廓、抗锯齿网点、排线与纸纹。
- 精细：在标准基础上，用原图灰度引导的局部线性系数重建底色，并重新提取原始分辨率的细结构；重建按带重叠边界的 512 图块执行，避免接缝。
- 网点间距以最长边 1000 为参考，随输出尺寸缩放；预览与导出色板一致。精细模式增加真实细节，所以与标准预览不要求逐像素相同。
- 分析图最长边 1000，属于传统多尺度抽象；不能恢复原图中不存在的细节，也不等同生成式重绘。
- 输入上限 1600 万像素、40 MB、单边 20000；超过时明确拒绝。透明度白底、EXIF 方向纠正、ICC 转为 sRGB，导出附 sRGB 配置。
- 解码缓存上限 64 MiB，中间层缓存上限 112 MiB；缓存按内容和影响阶段的参数区分，只调轮廓/网点可复用区域与色板。缓存跨任务复用，退出程序释放。
- 预览、导出和批处理共享 CPU 队列，另有应用资源锁；OpenCV 线程 2，OpenCL 关闭。输入和最终输出仍需驻留内存，缓存上限不等于进程总内存上限。

## 四种处理路线

| 风格 | 实际流程 |
|---|---|
| 经典卡通 | 双边滤波、自适应阈值、颜色层次、粗细轮廓融合 |
| 扁平插画 | Mean Shift 区域平滑、Lab 采样 K-Means、32³ 色彩查找表、局部空间投票、轻微线条；人脸保留少量连续渐变 |
| 美式漫画 | 明度分级、独立粗细轮廓、亮度控制网点半径、可调角度/间距/浓度、边缘抗锯齿 |
| 铅笔素描 | XDoG 风格软阈值线条、主要轮廓、随明暗变化的单向/交叉排线、确定性纸张颗粒、可选彩铅 |

人像检测使用 OpenCV 自带 Haar 级联，完全离线；不含人脸识别，不下载深度模型。对侧脸、小脸、遮挡和强逆光可能漏检。主要/内部轮廓按不同空间尺度提取，并不是语义意义上的精确外轮廓。

## 测试和评估命令

```powershell
Set-Location D:\CartoonifyStudio
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/pytest.xml
.\.venv\Scripts\python.exe smoke_test.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe benchmark_v2.py
.\.venv\Scripts\python.exe validate_delivery.py
Start-Process .\outputs\gallery.html
Start-Process .\outputs\blind_review.html
```

`evaluate.py` 处理 21 张正式样本的四种风格，生成 84 张 PNG、21 张五联对比、24 张新旧对照/盲评图，并对真实高清参考生成 4 张原尺寸效果和原生裁剪对照。`blind_review.html` 可选择 A/B/相当，评分仅保存在浏览器，可导出 JSON；映射答案另存 `blind_key.json`，程序不伪造用户评分。

`benchmark_v2.py` 对 640×480、1280×720、1920×1080、3840×2160 的新旧四风格使用相同真实高分辨率输入，独立进程预热后测量 3 次、记录中位数/P95 和 RSS；另测缓存调参和量化。只有 3 次的 P95 是描述性数字，不是长期服务尾延迟。

`outputs/timings.json` 有全部照片处理分阶段耗时、编码耗时、人像框、预览/导出差异以及细碎线条指标。指标是诊断工具，低碎线不自动等于艺术效果更好；人工盲评仍有必要。

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
