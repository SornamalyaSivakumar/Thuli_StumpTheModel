"""
Synthetic catalogue generator — STAND-IN for a real scrape.

Why this exists: the sandbox this was built in has no general internet
access (pip/npm/github only) and shooting real phone photos requires a
real camera, so this script procedurally renders a large set of visually
distinct "eyewear" product images to prove the retrieval pipeline end to
end. It is NOT the deliverable catalogue — see scraper_template.py for
the real scraper to run against a live retailer, on a machine with full
network access.

Each SKU is a combination of frame_shape x frame_color x lens_tint x
temple_style x size, giving thousands of visually distinct but
genuinely confusable items (which is the point: cheap near-duplicates
are what make retrieval hard, same as a real eyewear catalogue where
many SKUs differ only by color).
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFilter
from backend.paths import IMG_DIR as OUT_DIR, META_PATH

random.seed(42)

FRAME_SHAPES = ["round", "square", "cateye", "aviator", "rectangle", "hexagon"]
FRAME_COLORS = [
    (20, 20, 20), (120, 80, 40), (200, 50, 50), (40, 90, 160),
    (30, 130, 90), (180, 140, 20), (140, 30, 130), (90, 90, 90),
    (230, 200, 180), (10, 60, 60),
]
LENS_TINTS = [
    (60, 60, 60, 140), (30, 60, 100, 120), (100, 60, 30, 120),
    (20, 20, 20, 180), (200, 200, 200, 60),
]
TEMPLE_STYLES = ["thin", "thick", "patterned"]

CANVAS = 256


def rnd_jitter(seed, amp):
    r = random.Random(seed * 7919 + 3)
    return r.uniform(-amp, amp)


def rnd_int(seed, lo, hi):
    r = random.Random(seed * 104729 + 11)
    return r.randint(lo, hi)


def draw_glasses(frame_shape, frame_color, lens_tint, temple_style, scale):
    img = Image.new("RGB", (CANVAS, CANVAS), (250, 248, 245))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = CANVAS // 2, CANVAS // 2
    lens_w = int(70 * scale)
    lens_h = int(50 * scale)
    gap = int(20 * scale)
    lw = max(3, int(6 * scale))

    def lens_bbox(offset_x):
        x0 = cx + offset_x - lens_w // 2
        y0 = cy - lens_h // 2
        return [x0, y0, x0 + lens_w, y0 + lens_h]

    left_bbox = lens_bbox(-gap // 2 - lens_w // 2)
    right_bbox = lens_bbox(gap // 2 + lens_w // 2)

    def draw_lens(bbox):
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        if frame_shape == "round":
            draw.ellipse(bbox, outline=frame_color, width=lw, fill=lens_tint)
        elif frame_shape == "square":
            draw.rectangle(bbox, outline=frame_color, width=lw, fill=lens_tint)
        elif frame_shape == "cateye":
            pts = [(x0, y1), (x0, y0 + h * 0.3), (x1, y0), (x1, y1)]
            draw.polygon(pts, fill=lens_tint)
            draw.line(pts + [pts[0]], fill=frame_color, width=lw)
        elif frame_shape == "aviator":
            pts = [(x0, y0 + h * 0.2), (x1, y0), (x1, y1 - h * 0.1),
                   (cx, y1), (x0, y1 - h * 0.15)]
            draw.polygon(pts, fill=lens_tint)
            draw.line(pts + [pts[0]], fill=frame_color, width=lw)
        elif frame_shape == "rectangle":
            draw.rounded_rectangle(bbox, radius=6, outline=frame_color, width=lw, fill=lens_tint)
        elif frame_shape == "hexagon":
            pts = [(x0 + w * 0.25, y0), (x0 + w * 0.75, y0), (x1, y0 + h / 2),
                   (x0 + w * 0.75, y1), (x0 + w * 0.25, y1), (x0, y0 + h / 2)]
            draw.polygon(pts, fill=lens_tint)
            draw.line(pts + [pts[0]], fill=frame_color, width=lw)

    draw_lens(left_bbox)
    draw_lens(right_bbox)
    draw.line([(left_bbox[2], cy), (right_bbox[0], cy)], fill=frame_color, width=lw)

    temple_w = 2 if temple_style == "thin" else (5 if temple_style == "thick" else 3)
    for bbox, direction in [(left_bbox, -1), (right_bbox, 1)]:
        x = bbox[0] if direction < 0 else bbox[2]
        y = cy
        end_x = x + direction * 40 * scale
        end_y = y - 5
        if temple_style == "patterned":
            for i in range(4):
                t = i / 3
                px, py = x + (end_x - x) * t, y + (end_y - y) * t
                draw.line([(px, py - 1), (px, py + 1)], fill=frame_color, width=1)
        draw.line([(x, y), (end_x, end_y)], fill=frame_color, width=temple_w)

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def build_catalogue(n_items=5200):
    os.makedirs(OUT_DIR, exist_ok=True)
    combos = []
    for shape in FRAME_SHAPES:
        for color in FRAME_COLORS:
            for tint in LENS_TINTS:
                for temple in TEMPLE_STYLES:
                    combos.append((shape, color, tint, temple))
    random.Random(1).shuffle(combos)

    metadata = []
    i = 0
    while len(metadata) < n_items:
        shape, color, tint, temple = combos[i % len(combos)]
        scale = 1.0 + rnd_jitter(i, 0.15)
        jcolor = tuple(min(255, max(0, c + rnd_int(i, -18, 18))) for c in color)
        item_id = f"SKU{i:06d}"
        img = draw_glasses(shape, jcolor, tint, temple, scale)
        fname = f"{item_id}.jpg"
        img.save(os.path.join(OUT_DIR, fname), quality=90)
        metadata.append({
            "item_id": item_id,
            "file": fname,
            "frame_shape": shape,
            "frame_color_rgb": list(jcolor),
            "lens_tint": list(tint),
            "temple_style": temple,
            "scale": round(scale, 3),
        })
        i += 1
        if i % 1000 == 0:
            print(f"  {i}/{n_items}")

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)
    print(f"Generated {len(metadata)} catalogue items -> {OUT_DIR}")
    return metadata


if __name__ == "__main__":
    build_catalogue(n_items=5200)
