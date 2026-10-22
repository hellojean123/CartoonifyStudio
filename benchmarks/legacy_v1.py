from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image, ImageOps

cv2.setNumThreads(2)
MAX_PIXELS = 16_000_000
STYLES = ['经典卡通', '扁平插画', '美式漫画', '铅笔素描']
PRESETS = [(0.55, 5, 16, 1.05), (0.22, 7, 12, 1.0), (0.8, 4, 8, 1.35), (0.5, 5, 20, 1.1)]

@dataclass
class Params:
    outline: float = .55
    smooth: int = 5
    colors: int = 16
    contrast: float = 1.05
    colored: bool = False

def load_image(path):
    with Image.open(path) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError('图片超过 1600 万像素，请先缩小。')
        im = ImageOps.exif_transpose(im)
        if 'A' in im.getbands() or 'transparency' in im.info:
            rgba = im.convert('RGBA')
            bg = Image.new('RGBA', rgba.size, 'white')
            im = Image.alpha_composite(bg, rgba)
        return np.array(im.convert('RGB'))

def resize(image, limit):
    h, w = image.shape[:2]
    scale = min(1., limit / max(h, w))
    return cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

def quantize(im, colors):
    # Deterministic, bounded sample fitting; mapping is chunked to limit RAM.
    pixels = im.reshape(-1, 3)
    rng = np.random.default_rng(42)
    sample = pixels[rng.choice(len(pixels), min(10000, len(pixels)), replace=False)].astype(np.float32)
    k = min(colors, len(np.unique(sample, axis=0)))
    cv2.setRNGSeed(42)
    _, _, centers = cv2.kmeans(sample, k, None, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 25, .7), 1, cv2.KMEANS_PP_CENTERS)
    out = np.empty_like(pixels)
    for start in range(0, len(pixels), 32768):
        chunk = pixels[start:start+32768].astype(np.float32)
        distances = ((chunk[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        out[start:start+len(chunk)] = centers[distances.argmin(axis=1)]
    return out.reshape(im.shape)

def process(image, style, params=None, preview=False):
    if style not in STYLES:
        raise ValueError('未知风格')
    p = params or Params(*PRESETS[STYLES.index(style)])
    edge = float(np.clip(p.outline, 0, 1))
    smooth = int(np.clip(p.smooth, 1, 10))
    colors = int(np.clip(p.colors, 4, 48))
    contrast = float(np.clip(p.contrast, .6, 1.8))
    im = np.asarray(image)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if im.ndim != 3 or im.shape[2] != 3 or im.size == 0 or im.dtype != np.uint8:
        raise ValueError('需要非空 uint8 RGB 图像')
    if im.shape[0]*im.shape[1] > MAX_PIXELS:
        raise ValueError('图片超过 1600 万像素')
    im = resize(im, 1000) if preview else im.copy()
    im = np.clip((im.astype(np.float32)-127.5)*contrast+127.5, 0, 255).astype(np.uint8)
    # All expensive neighborhood operations use a bounded working resolution.
    work = resize(im, 1400)
    blur = cv2.bilateralFilter(work, 5+2*(smooth//3), 25+smooth*6, 8+smooth*3)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    if style == STYLES[0]:
        mask = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
        base = cv2.bilateralFilter(blur, 7, 80, 25)
        graded = np.round(base.astype(np.float32)/12)*12
        base = np.clip(base*.35+graded*.65, 0, 255)
        factor = 1-edge*.78*(1-mask.astype(np.float32)/255)
        result = (base*factor[..., None]).astype(np.uint8)
    elif style == STYLES[1]:
        base = quantize(cv2.medianBlur(blur, 5), colors)
        lines = cv2.Canny(gray, 85, 180).astype(np.float32)/255
        result = (base*(1-lines[..., None]*edge*.55)).astype(np.uint8)
    elif style == STYLES[2]:
        levels = max(4, min(12, colors))
        comic = cv2.bilateralFilter(cv2.medianBlur(blur, 5), 7, 90, 30)
        hsv = cv2.cvtColor(comic, cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32)*1.08, 0, 255).astype(np.uint8)
        hsv[..., 2] = np.clip(np.round(hsv[..., 2].astype(np.float32)/255*(levels-1))/(levels-1)*255, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        comic_gray = cv2.GaussianBlur(cv2.cvtColor(comic, cv2.COLOR_RGB2GRAY), (0, 0), 1.2)
        lines = cv2.Canny(comic_gray, 85, 170)
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        yy, xx = np.indices(gray.shape)
        radius = np.clip((175-gray.astype(np.float32))/175, 0, 1)*2.6
        dots = ((xx % 7-3)**2+(yy % 7-3)**2) < radius**2
        factor = (1-dots*.42)*(1-lines/255*edge*.94)
        result = np.clip(base*factor[..., None], 0, 255).astype(np.uint8)
    else:
        inv = 255-gray
        soft = cv2.GaussianBlur(inv, (0, 0), 3+smooth*1.5)
        sketch = cv2.divide(gray, 255-soft, scale=256)
        sketch[(gray < 8) & (soft > 247)] = 255
        gradient = cv2.Laplacian(gray, cv2.CV_32F)
        sketch = np.clip(sketch.astype(np.float32)-np.abs(gradient)*edge*.45, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        if p.colored:
            result = np.clip(result.astype(np.float32)*(.45+.55*blur.astype(np.float32)/255), 0, 255).astype(np.uint8)
    if result.shape != im.shape:
        # Restore original dimensions and retain a little original fine detail.
        result = cv2.resize(result, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_CUBIC)
    return result
