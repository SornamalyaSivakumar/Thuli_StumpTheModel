"""
Embedding function used by both catalogue indexing and query-time matching.

IMPORTANT — read this before judging retrieval quality:
This sandbox has no general internet access (only pypi/npm/github are
reachable) and installing PyTorch here blew the disk quota (CPU wheels
pull in >2GB of bundled CUDA libraries even for "CPU-only" installs from
PyPI, and the box only had ~5GB free). So this demo uses a classical
computer-vision embedding: a multi-region HSV color histogram + a HOG
shape descriptor, concatenated and L2-normalized. It is real, it runs,
and it is honestly weaker than a learned embedding — in particular it is
far more sensitive to lighting/color shifts than a CNN or CLIP embedding
would be, which shows up directly in the Part B results.

To run this for real: swap `embed_image` for a CLIP forward pass
(e.g. via `open_clip`, ViT-B/32, laion2b weights) — everything else
(FAISS index, matcher, API, eval harness) is agnostic to what produces
the vector, as long as `embed_image` returns a fixed-length np.float32
vector and `EMBED_DIM` matches. That's the one function to change.
"""
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

RESIZE = 128
HIST_GRID = 4        # 4x4 spatial grid for color histogram
H_BINS, S_BINS, V_BINS = 8, 4, 4
HOG_RESIZE = 64

COLOR_DIM = HIST_GRID * HIST_GRID * (H_BINS + S_BINS + V_BINS)
# hog output dim computed lazily at first call and cached
_HOG_DIM = None
EMBED_DIM = None  # set after first call via get_embed_dim()


def _color_hist(img_rgb: np.ndarray) -> np.ndarray:
    """Multi-region HSV histogram — captures coarse color + spatial layout."""
    from colorsys import rgb_to_hsv
    h, w, _ = img_rgb.shape
    cell_h, cell_w = h // HIST_GRID, w // HIST_GRID
    feats = []
    hsv = np.array(Image.fromarray(img_rgb).convert("HSV"), dtype=np.float32) / 255.0
    for gy in range(HIST_GRID):
        for gx in range(HIST_GRID):
            cell = hsv[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            hh, _ = np.histogram(cell[:, :, 0], bins=H_BINS, range=(0, 1))
            sh, _ = np.histogram(cell[:, :, 1], bins=S_BINS, range=(0, 1))
            vh, _ = np.histogram(cell[:, :, 2], bins=V_BINS, range=(0, 1))
            cell_hist = np.concatenate([hh, sh, vh]).astype(np.float32)
            total = cell_hist.sum()
            if total > 0:
                cell_hist /= total
            feats.append(cell_hist)
    return np.concatenate(feats)


def _hog_feat(img_rgb: np.ndarray) -> np.ndarray:
    small = np.array(Image.fromarray(img_rgb).resize((HOG_RESIZE, HOG_RESIZE)))
    gray = rgb2gray(small)
    feat = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               feature_vector=True)
    return feat.astype(np.float32)


def _autocrop_foreground(pil_img: Image.Image, bg_thresh: int = 235) -> Image.Image:
    """Crops to the bounding box of non-near-white pixels, then pads back to
    square. Mitigation attempt aimed at rotation / off-angle-crop / cluttered
    background: those conditions push the item off-center or shrink it
    relative to the frame, which both the color-grid and HOG features are
    sensitive to. Cheap, no model required — but only helps when the
    background genuinely differs from the foreground in brightness, which
    is exactly the assumption that breaks under cluttered_background."""
    arr = np.array(pil_img.convert("RGB"))
    mask = np.any(arr < bg_thresh, axis=2)
    if not mask.any():
        return pil_img
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cropped = pil_img.crop((x0, y0, x1 + 1, y1 + 1))
    w, h = cropped.size
    side = max(w, h)
    padded = Image.new("RGB", (side, side), (250, 248, 245))
    padded.paste(cropped, ((side - w) // 2, (side - h) // 2))
    return padded


def embed_image(pil_img: Image.Image, autocrop: bool = False) -> np.ndarray:
    """Returns a single L2-normalized float32 embedding vector for one image.
    autocrop=True applies the foreground-cropping mitigation (see above) —
    off by default so the baseline number is a clean, unmitigated result."""
    global EMBED_DIM
    if autocrop:
        pil_img = _autocrop_foreground(pil_img)
    img = pil_img.convert("RGB").resize((RESIZE, RESIZE))
    arr = np.array(img)
    color = _color_hist(arr)
    shape = _hog_feat(arr)
    vec = np.concatenate([color, shape]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    if EMBED_DIM is None:
        EMBED_DIM = vec.shape[0]
    return vec


def get_embed_dim() -> int:
    global EMBED_DIM
    if EMBED_DIM is None:
        # probe with a blank image
        embed_image(Image.new("RGB", (RESIZE, RESIZE), (255, 255, 255)))
    return EMBED_DIM
