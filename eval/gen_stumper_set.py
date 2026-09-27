"""
Builds the AUTOMATED stumper set (the 'take it further' extension):
samples catalogue items, applies one or two stacked perturbations per
image, and records ground truth + the exact condition labels applied.

This is separate from, and complementary to, the hand-shot phone set
(eval/handshot_labels.csv, which a human fills in after shooting real
photos — see eval/README.md). The point of building this one is to show
it defeats the matcher at a HIGHER rate than the hand-shot set, since it
can stack conditions and push them harder than a phone snapshot can.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from eval.perturb import apply, CONDITIONS
from backend.paths import (CAT_DIR, IMG_DIR, META_PATH,
                            STUMPER_AUTO_DIR as OUT_DIR,
                            STUMPER_AUTO_LABELS as LABELS_PATH)

HARD_CONDITIONS = [c for c in CONDITIONS if c != "clean"]


def build(n_single=150, n_double=150, seed=7):
    r = random.Random(seed)
    with open(META_PATH) as f:
        metadata = json.load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    labels = []
    idx = 0

    # single-condition cases
    for _ in range(n_single):
        item = r.choice(metadata)
        cond = r.choice(HARD_CONDITIONS)
        img = Image.open(os.path.join(IMG_DIR, item["file"]))
        out = apply(img, cond, seed=idx)
        fname = f"auto_{idx:04d}.jpg"
        out.convert("RGB").save(os.path.join(OUT_DIR, fname), quality=85)
        labels.append({"file": fname, "true_item_id": item["item_id"],
                        "conditions": [cond]})
        idx += 1

    # double-condition (stacked) cases — the genuinely nasty ones
    for _ in range(n_double):
        item = r.choice(metadata)
        c1, c2 = r.sample(HARD_CONDITIONS, 2)
        img = Image.open(os.path.join(IMG_DIR, item["file"]))
        mid = apply(img, c1, seed=idx)
        out = apply(mid, c2, seed=idx + 9999)
        fname = f"auto_{idx:04d}.jpg"
        out.convert("RGB").save(os.path.join(OUT_DIR, fname), quality=85)
        labels.append({"file": fname, "true_item_id": item["item_id"],
                        "conditions": sorted([c1, c2])})
        idx += 1

    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Built {len(labels)} automated stumper images -> {OUT_DIR}")
    print(f"  {n_single} single-condition, {n_double} double-condition (stacked)")


if __name__ == "__main__":
    build()
