"""
Programmatic 'hard photo' generator: takes a clean catalogue image and
applies a named perturbation simulating one real-world failure condition.
Used by (a) the automated stumper set and (b) threshold calibration.
"""
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

CONDITIONS = [
    "clean", "low_light", "overexposed", "motion_blur", "rotation",
    "partial_occlusion", "cluttered_background", "reflection_glare",
    "off_angle_crop", "low_res",
]


def _rng(seed):
    return random.Random(seed)


def apply(img: Image.Image, condition: str, seed=0) -> Image.Image:
    img = img.convert("RGB")
    r = _rng(seed)

    if condition == "clean":
        return img

    if condition == "low_light":
        return ImageEnhance.Brightness(img).enhance(r.uniform(0.25, 0.4))

    if condition == "overexposed":
        return ImageEnhance.Brightness(img).enhance(r.uniform(1.8, 2.3))

    if condition == "motion_blur":
        angle_img = img.rotate(r.uniform(-3, 3), expand=False, fillcolor=(250, 248, 245))
        return angle_img.filter(ImageFilter.GaussianBlur(r.uniform(4, 7)))

    if condition == "rotation":
        return img.rotate(r.uniform(25, 50) * r.choice([-1, 1]),
                           expand=True, fillcolor=(250, 248, 245))

    if condition == "partial_occlusion":
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        # simulate a finger/thumb covering part of the item
        ox, oy = r.randint(0, w // 2), r.randint(0, h // 2)
        ow, oh = int(w * r.uniform(0.35, 0.55)), int(h * r.uniform(0.35, 0.55))
        skin = (r.randint(150, 210), r.randint(100, 160), r.randint(90, 140))
        draw.ellipse([ox, oy, ox + ow, oy + oh], fill=skin)
        return img

    if condition == "cluttered_background":
        w, h = img.size
        bg = Image.new("RGB", (w, h))
        bgdraw = ImageDraw.Draw(bg)
        for _ in range(40):
            x0, y0 = r.randint(0, w), r.randint(0, h)
            x1, y1 = x0 + r.randint(5, 30), y0 + r.randint(5, 30)
            color = (r.randint(0, 255), r.randint(0, 255), r.randint(0, 255))
            bgdraw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=color)
        bg = bg.filter(ImageFilter.GaussianBlur(2))
        # paste item scaled down in a random position, rest is clutter
        scale = r.uniform(0.5, 0.7)
        small = img.resize((int(w * scale), int(h * scale)))
        px = r.randint(0, w - small.width)
        py = r.randint(0, h - small.height)
        bg.paste(small, (px, py))
        return bg

    if condition == "reflection_glare":
        img = img.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        x0 = r.randint(0, w // 2)
        draw.polygon([(x0, 0), (x0 + w // 3, 0), (x0 - w // 4, h), (x0 - w // 2, h)],
                     fill=(255, 255, 255, r.randint(90, 150)))
        return img

    if condition == "off_angle_crop":
        w, h = img.size
        # simulate shooting at a steep angle: skew via perspective-ish crop + squeeze
        crop = img.crop((int(w * 0.1), int(h * 0.15), w, int(h * 0.9)))
        return crop.resize((w, h)).transform(
            (w, h), Image.AFFINE, (1, r.uniform(0.2, 0.35), 0, 0, 1, 0),
            fillcolor=(250, 248, 245))

    if condition == "low_res":
        w, h = img.size
        tiny = img.resize((max(16, w // 8), max(16, h // 8)))
        return tiny.resize((w, h))

    raise ValueError(f"unknown condition {condition}")
