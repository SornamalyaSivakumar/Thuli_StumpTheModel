"""
Calibrates the two open-set rejection thresholds (top-1 similarity,
top1-top2 margin) by sweeping candidate values against:
  - POSITIVES: automated stumper queries (perturbed catalogue images) —
    these SHOULD be accepted (matched) since the true item is genuinely
    in the catalogue.
  - NEGATIVES: non-catalogue distractor images — these SHOULD be
    rejected (no_match) since the true item is genuinely absent.

Reports False Accept Rate (negative wrongly accepted) and False Reject
Rate (positive wrongly rejected) at each threshold pair, and picks the
pair minimizing FAR+FRR as the default operating point (a simple,
defensible choice — a deployed system would instead pick a point on the
curve matching its actual cost tradeoff, e.g. biasing toward low FAR if
false accepts are more expensive than false rejects).
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from backend.matcher import Matcher
from backend.catalogue_meta import load_metadata_by_id
from backend.paths import (CAT_DIR, INDEX_FLAT_PATH, IDS_PATH, EVAL_DIR,
                            STUMPER_AUTO_DIR as STUMPER_DIR,
                            STUMPER_AUTO_LABELS as STUMPER_LABELS,
                            NONCAT_DIR, CALIBRATED_THRESHOLDS_PATH)

SIM_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MARGIN_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]


def main():
    meta = load_metadata_by_id()
    # use raw (uncalibrated) thresholds just to get raw top1/margin numbers for every query,
    # then sweep thresholds in-memory rather than re-running search per threshold pair.
    m = Matcher(INDEX_FLAT_PATH, IDS_PATH, meta,
                sim_threshold=-1, margin_threshold=-1)  # accept everything, just want raw scores

    with open(STUMPER_LABELS) as f:
        pos_labels = json.load(f)

    pos_scores = []  # (top1, margin) for genuine in-catalogue queries
    for lab in pos_labels:
        img = Image.open(os.path.join(STUMPER_DIR, lab["file"]))
        r = m.match(img)
        pos_scores.append((r["top1_similarity"], r["margin"]))

    neg_scores = []  # (top1, margin) for genuinely absent items
    for fname in sorted(os.listdir(NONCAT_DIR)):
        img = Image.open(os.path.join(NONCAT_DIR, fname))
        r = m.match(img)
        neg_scores.append((r["top1_similarity"], r["margin"]))

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    print(f"Positives (in-catalogue, n={len(pos_scores)}): "
          f"top1 mean={pos_scores[:,0].mean():.3f} median={np.median(pos_scores[:,0]):.3f} | "
          f"margin mean={pos_scores[:,1].mean():.3f} median={np.median(pos_scores[:,1]):.3f}")
    print(f"Negatives (non-catalogue, n={len(neg_scores)}): "
          f"top1 mean={neg_scores[:,0].mean():.3f} median={np.median(neg_scores[:,0]):.3f} | "
          f"margin mean={neg_scores[:,1].mean():.3f} median={np.median(neg_scores[:,1]):.3f}")

    best = None
    results = []
    for sim_t, margin_t in itertools.product(SIM_GRID, MARGIN_GRID):
        accepted_pos = (pos_scores[:, 0] >= sim_t) & (pos_scores[:, 1] >= margin_t)
        accepted_neg = (neg_scores[:, 0] >= sim_t) & (neg_scores[:, 1] >= margin_t)
        frr = 1 - accepted_pos.mean()          # genuine items wrongly rejected
        far = accepted_neg.mean()               # absent items wrongly accepted
        score = far + frr
        results.append((sim_t, margin_t, far, frr, score))
        if best is None or score < best[4]:
            best = (sim_t, margin_t, far, frr, score)

    results.sort(key=lambda x: x[4])
    print("\nTop 8 threshold pairs by FAR+FRR:")
    print(f"{'sim_t':>6} {'margin_t':>9} {'FAR':>7} {'FRR':>7} {'FAR+FRR':>8}")
    for sim_t, margin_t, far, frr, score in results[:8]:
        print(f"{sim_t:6.2f} {margin_t:9.3f} {far:7.3f} {frr:7.3f} {score:8.3f}")

    print(f"\n==> Selected operating point: sim_threshold={best[0]}, "
          f"margin_threshold={best[1]}  (FAR={best[2]:.3f}, FRR={best[3]:.3f})")

    with open(CALIBRATED_THRESHOLDS_PATH, "w") as f:
        json.dump({"sim_threshold": best[0], "margin_threshold": best[1],
                   "far": best[2], "frr": best[3]}, f, indent=2)


if __name__ == "__main__":
    main()
