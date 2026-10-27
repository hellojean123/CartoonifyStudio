# 铅笔素描 v5：明暗与笔触改进实测

日期：2026-10-27。项目：`D:\CartoonifyStudio`。CPU OpenCV 2 线程，OpenCL 关闭；素描明暗和笔触不使用 GPU、风格生成模型或外部纹理；自动人像定位沿用已有的本地 CPU 人脸检测模型。Python 3.12.10。

## 问题和实际修改

v4 虽有清楚轮廓，但随机噪声沿局部方向拉伸会产生卷曲杂纹；暗部依赖局部方差，平滑阴影容易被抹掉。统一的强黑边也容易把眼睛画成黑圈。

1. 新建独立 `cartoonify/pencil.py`，分别生成结构线、明暗密度、有限长度笔触和颜色吸收系数。算法与 Gradio 保持分离。
2. 多尺度结构线结合方向一致性和连通长度筛选；沿逐步更新的方向场积分，减少孤立细碎边缘。轮廓浓度用连续吸收控制，避免统一的纯黑描边。
3. 明暗由平滑亮度生成，不再要求暗色均匀区域必须变成白纸。每张脸分别估计亮部参考，保留浅肤色人像的鼻侧、面颊等宽缓阴影；不会把皮肤纹理作为明暗来源。
4. 使用有长度、方向变化与分段压力的短笔触，暗部加入第二方向排线。通过量化压力组批量绘制，替代逐笔三角函数和绘图循环。固定种子使结果可复现。
5. 纸纹通过稳定坐标哈希调制石墨附着，不再覆盖全局周期正弦纹。亮部无颜料时仍为白纸；彩铅的颜色来自笔触吸收，不是给照片染纸底。
6. 皮肤细线和背景线条使用柔和区域权重；人像保护开关、力度和手动框仍可用。颜色、排线、纸纹及轮廓在渲染阶段调整，可复用结构缓存。
7. 细笔触加入适度抗锯齿，高清层采样改用双三次；分块坐标保持一致。移除了旧版随机流噪声代码。

本实现借鉴论文的分层思想与流向滤波，并非完整复现论文中的方向分类、纹理优化或物理铅笔模型。没有复制第三方铅笔实现或下载第三方纸纹。

## 可核验输出

- [21 张新旧 / OpenCV 对照画廊](outputs/pencil_v5/gallery.html)：原图、准确的已发布 v4 图像、新版、OpenCV 固定参数基线。
- 新版基线固定为 Git 提交 `0a2cb624facc1dd3bca249c1ec9df421d8c71544` 中的图像，不用新环境重跑旧算法冒充旧结果。
- `outputs/pencil_v5/`：21 张原尺寸素描、21 张原尺寸彩铅、21 张对照图、21 张原始像素裁剪对照、21 张 OpenCV 基线及总览。
- `outputs/4_铅笔素描/`：当前默认黑白素描；`outputs/*_comparison.jpg` 与 `outputs/gallery.html` 已更新。
- `outputs/highres_4.png`：真实高清参考照片的 3840×2880 PNG；[来源署名](samples/high_resolution/ATTRIBUTION.md)。
- [逐图耗时与裁剪坐标](outputs/pencil_v5/evaluation.json)、[其他风格哈希核验](outputs/pencil_v5/unchanged_other_styles.json)。另三种风格的 63 张结果保持原哈希，拼豆算法未修改。

OpenCV 对照参数为 `sigma_s=60, sigma_r=.07, shade_factor=.05`，在最长边 1000 输入上运行，保留其自然灰度输出。它用于呈现不同实现的差别，不能代表 OpenCV 最佳调参。新版原尺寸计时与此基线输入尺寸不同，不能直接据此计算加速比。

## 测试与性能

- 完整 pytest：**96 通过，0 失败，129.96 秒**。[XML](outputs/pencil_v5/pytest.xml)
- 新增检查：平滑阴影明暗、亮肤色面部层次、真实短笔触可复现、彩铅亮部留白、渲染参数复用缓存、高清分块无接缝及预览一致性。旧测试“所有平坦暗区域强制白纸”已改为检查暗部具有明暗层次；排线设为 0 时，均匀场仍为空白纸。
- 实际 Gradio HTTP：两阶段预览、原尺寸黑白 PNG、原尺寸彩铅 JPG、ICC 色彩配置、两张批量 PNG ZIP 及取消入口通过。[HTTP 记录](outputs/pencil_v5/http_smoke.json)
- 全部 21 张原尺寸素描与预览重采样差异：平均 **1.699/255**，最大 **2.501/255**；历史 v4 同一批记录最大为 **12.539/255**。这是像素一致性指标，不是审美评分。

独立子进程以同一张授权人像按比例缩放，缓存关闭处理 3 次取中位数；参数预览先预热结构缓存。输入加载、PNG/JPG 编码、HTTP 上传及队列等待不计入下表算法耗时。RSS 为进程总内存，每 10 ms 采样，包含 Python/OpenCV、输入、输出和预热缓存，不是精确的峰值分配器读数。

| 输入宽×高 | 无缓存处理中位数 | 命中缓存后调明暗预览 | 峰值进程 RSS |
| --- | --- | --- | --- |
| 457×640 | 0.528 秒 | 0.041 秒 | 103.4 MiB |
| 771×1080 | 1.183 秒 | 0.351 秒 | 156.5 MiB |
| 2743×3840 | 7.443 秒 | 0.529 秒 | 186.0 MiB |

最大人像行最长边为 3840，保持原人像比例；另外单独生成了 3840×2880 风景导出。3 次具体耗时、首次与后续运行、阶段时间均见 [原始性能数据](outputs/pencil_v5/benchmark.json)。本机其他负载会影响速度，不能据此保证所有电脑都相同。

## 目视检查与局限

查看了全部 21 张总览，并放大核验老人、儿童、侧脸、浅肤色官方人像、多人户外、建筑等代表图。第二轮修复了灰底过重，之后又修复了浅肤色面部过白；最终图保留了比 v4 更完整的明暗，并减少卷曲杂纹。此处是工程目视检查，没有伪造用户盲评或审美分数。

- 方向场来自二维照片，不知道真实三维表面，仍可能出现局部方向不合适或排线偏密。
- 复杂树叶、小脸、老人皱纹仍可能有碎线；背景简化依赖柔和人脸邻域，不是精确的人物分割。
- 深色平坦背景现在会铺石墨明暗。如果希望大面积留白，可把“铅笔明暗与排线浓度”从默认 0.48 降到约 0.25，设为 0 得到纯结构线。
- 高清导出复用最长边 1000 的笔触布局，不生成额外高清笔画；极大放大仍会看到重采样痕迹。
- 彩铅是传统色彩吸收近似，没有真实颜料混合或完整铅笔物理仿真。

## 研究依据

- Lu / Xu / Jia, NPAR 2012：[Combining Sketch and Tone for Pencil Drawing Production](https://www.cse.cuhk.edu.hk/leojia/projects/pencilsketch/npar12_pencil.pdf)。借鉴结构线与明暗分层思想。
- Winnemöller / Kyprianidis / Olsen, 2012：[XDoG](https://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf)。借鉴流向一致性与连续线条控制思想。
- [OpenCV 4.12 官方文档](https://docs.opencv.org/4.12.0/df/dac/group__photo__render.html) 与 [Domain_Filter 源码](https://raw.githubusercontent.com/opencv/opencv/4.12.0/modules/photo/src/npr.hpp)。用作可复现的现成基线。

## 重现

```powershell
Set-Location D:\CartoonifyStudio
.\.venv\Scripts\python.exe evaluate_pencil_v5.py
.\.venv\Scripts\python.exe -m pytest -q --junitxml=outputs/pencil_v5/pytest.xml
.\.venv\Scripts\python.exe smoke_pencil_v5.py
```

对照脚本需要完整 Git 历史中的上述 v4 提交；普通运行应用不需要 Git 历史。照片许可与署名沿用 `samples/metadata.json` 和 `samples/ATTRIBUTION.md`；没有下载额外照片或未核实来源的材质。
