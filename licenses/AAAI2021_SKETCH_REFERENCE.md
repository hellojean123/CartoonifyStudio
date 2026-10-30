# AAAI 2021 sketch reference

Tong, Zhengyan; Chen, Xuanhong; Ni, Bingbing; Wang, Xiaohang. Sketch Generation with Drawing Process Guided by Vector Flow and Grayscale. AAAI 35(1), 609–616, 2021.

- Paper: https://ojs.aaai.org/index.php/AAAI/article/view/16140
- Official source: https://github.com/TZYSJTU/Sketch-Generation-with-Drawing-Process-Guided-by-Vector-Flow-and-Grayscale
- Official repository license verified 2026-10-30: CC0 1.0; copy in AAAI2021_SKETCH_CC0.txt.
- Inspected source URL and SHA256: outputs/pencil_v11/references.json.

cartoonify/flow_pencil.py is an independently written CPU adaptation using ETF, cumulative gray layers, quantized directions, interval strokes and importance ordering. It does not reproduce the original brush patches, full-canvas rotation per stroke, default CLAHE or numerical outputs. Existing pressure brush and portrait protection are integrated. No official example photos or generated results are redistributed.
