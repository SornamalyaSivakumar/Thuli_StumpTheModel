"""
Core retrieval logic.

Confidence & open-set rejection design
---------------------------------------
Raw top-1 cosine similarity is a poor confidence signal on its own: a
photo of an item that ISN'T in the catalogue still gets a "top-1" match
(FAISS always returns something), often with a deceptively high absolute
score if the embedding space is at all smooth. The signal that actually
separates "genuine match" from "forced nearest wrong guess" is the
MARGIN between top-1 and top-2: a real match tends to sit in a tight
cluster of near-duplicate variants (same SKU photographed differently,
or visually adjacent SKUs) and pulls clearly ahead of the pack; an
out-of-catalogue query has no such cluster to pull ahead of, so top-1
and top-2 land close together.

We use both signals:
  - top-1 similarity thresholded (catches "nothing in the catalogue is
    even vaguely similar")
  - top1-top2 margin thresholded (catches "everything looks similarly
    plausible / nothing stands out")
and reject (return NO_MATCH) if either fires. Both thresholds are
calibrated empirically against a held-out set of known non-catalogue
queries — see eval/calibrate_threshold.py — not hand-picked.
"""
import time
import numpy as np
import faiss

from .embedder import embed_image

SIM_THRESHOLD_DEFAULT = 0.35      # top-1 cosine sim below this -> reject
MARGIN_THRESHOLD_DEFAULT = 0.015  # top1-top2 gap below this -> reject


class Matcher:
    def __init__(self, index_path, ids_path, metadata_by_id,
                 sim_threshold=SIM_THRESHOLD_DEFAULT,
                 margin_threshold=MARGIN_THRESHOLD_DEFAULT,
                 top_k=5):
        self.index = faiss.read_index(index_path)
        import json
        with open(ids_path) as f:
            self.ids = json.load(f)
        self.metadata_by_id = metadata_by_id
        self.sim_threshold = sim_threshold
        self.margin_threshold = margin_threshold
        self.top_k = top_k

    def match(self, pil_img, k=None, autocrop=False):
        k = k or self.top_k
        search_k = max(k, 2)  # need at least 2 for margin calc

        t0 = time.perf_counter()
        vec = embed_image(pil_img, autocrop=autocrop).reshape(1, -1).astype(np.float32)
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        sims, idxs = self.index.search(vec, search_k)
        search_ms = (time.perf_counter() - t1) * 1000

        sims, idxs = sims[0], idxs[0]
        valid = idxs >= 0
        sims, idxs = sims[valid], idxs[valid]

        candidates = []
        for sim, idx in zip(sims[:k], idxs[:k]):
            item_id = self.ids[idx]
            meta = self.metadata_by_id.get(item_id, {})
            candidates.append({
                "item_id": item_id,
                "similarity": float(sim),
                "meta": meta,
            })

        top1 = float(sims[0]) if len(sims) > 0 else -1.0
        top2 = float(sims[1]) if len(sims) > 1 else -1.0
        margin = top1 - top2

        is_match = (top1 >= self.sim_threshold) and (margin >= self.margin_threshold)

        return {
            "candidates": candidates,
            "top1_similarity": top1,
            "margin": margin,
            "is_match": bool(is_match),
            "reason": None if is_match else self._reject_reason(top1, margin),
            "timing_ms": {
                "embed": round(embed_ms, 2),
                "search": round(search_ms, 2),
                "total": round(embed_ms + search_ms, 2),
            },
        }

    def _reject_reason(self, top1, margin):
        if top1 < self.sim_threshold:
            return f"top-1 similarity {top1:.3f} below threshold {self.sim_threshold} " \
                   f"— nothing in the catalogue looks like this"
        return f"margin {margin:.3f} below threshold {self.margin_threshold} " \
               f"— several catalogue items look equally (un)likely, no confident winner"
