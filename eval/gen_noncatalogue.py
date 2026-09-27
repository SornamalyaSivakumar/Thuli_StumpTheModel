"""
Generates objects that are genuinely NOT in the catalogue (different
object categories entirely: watches, rings, bracelets, bowties, masks)
to test open-set rejection — stand-in for "photograph 20 items you know
aren't in the catalogue" since this sandbox has no camera access.
Some are drawn deliberately eyewear-ADJACENT (e.g. a swim-goggle shape)
to make rejection genuinely hard, not just an easy sanity check.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFilter
from backend.paths import NONCAT_DIR as OUT_DIR

CANVAS = 256


def draw_watch(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    d = ImageDraw.Draw(img)
    cx, cy = CANVAS // 2, CANVAS // 2
    rad = r.randint(35, 50)
    color = (r.randint(0, 100), r.randint(0, 100), r.randint(0, 100))
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=6, fill=(230, 230, 230))
    d.rectangle([cx - 15, 20, cx + 15, cy - rad], fill=(60, 40, 30))
    d.rectangle([cx - 15, cy + rad, cx + 15, CANVAS - 20], fill=(60, 40, 30))
    for a in range(0, 360, 30):
        import math
        x = cx + (rad - 8) * math.cos(math.radians(a))
        y = cy + (rad - 8) * math.sin(math.radians(a))
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 0, 0))
    return img


def draw_ring(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    d = ImageDraw.Draw(img)
    cx, cy = CANVAS // 2, CANVAS // 2
    rad = r.randint(30, 45)
    color = (r.randint(150, 255), r.randint(150, 255), r.randint(0, 100))
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=10)
    gem = (r.randint(0, 255), r.randint(0, 255), r.randint(200, 255))
    d.polygon([(cx, cy - rad - 15), (cx - 12, cy - rad + 2), (cx + 12, cy - rad + 2)], fill=gem)
    return img


def draw_bracelet(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    d = ImageDraw.Draw(img)
    color = (r.randint(100, 200), r.randint(100, 200), r.randint(100, 200))
    y = CANVAS // 2
    for x in range(40, CANVAS - 40, 20):
        d.ellipse([x, y - 15, x + 18, y + 15], outline=color, width=4)
    return img


def draw_bowtie(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    d = ImageDraw.Draw(img)
    cx, cy = CANVAS // 2, CANVAS // 2
    color = (r.randint(0, 255), r.randint(0, 150), r.randint(0, 150))
    d.polygon([(cx - 50, cy - 25), (cx, cy), (cx - 50, cy + 25)], fill=color)
    d.polygon([(cx + 50, cy - 25), (cx, cy), (cx + 50, cy + 25)], fill=color)
    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(20, 20, 20))
    return img


def draw_swim_goggles(seed):
    """Deliberately eyewear-ADJACENT — the hard negative case."""
    r = random.Random(seed)
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    d = ImageDraw.Draw(img)
    cx, cy = CANVAS // 2, CANVAS // 2
    color = (r.randint(0, 150), r.randint(150, 255), r.randint(150, 255))
    for dx in [-45, 45]:
        d.ellipse([cx + dx - 30, cy - 30, cx + dx + 30, cy + 30], outline=color, width=10, fill=(180, 220, 230, 120))
    d.line([(cx - 15, cy - 40), (cx - 60, cy - 60)], fill=color, width=6)
    d.line([(cx + 15, cy - 40), (cx + 60, cy - 60)], fill=color, width=6)
    return img.filter(ImageFilter.GaussianBlur(0.4))


GENERATORS = [draw_watch, draw_ring, draw_bracelet, draw_bowtie, draw_swim_goggles]


def build(n=20):
    os.makedirs(OUT_DIR, exist_ok=True)
    for i in range(n):
        gen = GENERATORS[i % len(GENERATORS)]
        img = gen(seed=i * 31 + 7)
        img.save(os.path.join(OUT_DIR, f"noncat_{i:03d}.jpg"), quality=88)
    print(f"Generated {n} non-catalogue distractor images -> {OUT_DIR}")


if __name__ == "__main__":
    build(20)
