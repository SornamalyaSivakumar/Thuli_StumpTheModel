"""
Embeds the whole catalogue offline and builds two FAISS indices:
  - flat (exact, brute-force cosine via inner product on normalized vecs)
  - IVF-PQ (approximate, for the "how fast / how far can this scale" story)

Run once after gen_catalogue.py. Saves:
  catalogue/embeddings.npy   (N x D float32, L2-normalized)
  catalogue/ids.json         (ordered list of item_id matching embeddings.npy rows)
  catalogue/index_flat.faiss
  catalogue/index_ivfpq.faiss
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from PIL import Image
from backend.embedder import embed_image, get_embed_dim
from backend.paths import (CAT_DIR, IMG_DIR, META_PATH, EMBEDDINGS_PATH, IDS_PATH,
                            INDEX_FLAT_PATH, INDEX_IVFPQ_PATH)


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    dim = get_embed_dim()
    n = len(metadata)
    embs = np.zeros((n, dim), dtype=np.float32)
    ids = []

    t0 = time.time()
    for i, item in enumerate(metadata):
        img = Image.open(os.path.join(IMG_DIR, item["file"]))
        embs[i] = embed_image(img)
        ids.append(item["item_id"])
        if (i + 1) % 1000 == 0:
            print(f"  embedded {i+1}/{n}  ({(time.time()-t0):.1f}s elapsed)")
    print(f"Embedding all {n} items took {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/n*1000:.2f} ms/item)")

    np.save(EMBEDDINGS_PATH, embs)
    with open(IDS_PATH, "w") as f:
        json.dump(ids, f)

    # --- flat exact index (cosine sim via inner product, vectors are L2-normed) ---
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(embs)
    faiss.write_index(index_flat, INDEX_FLAT_PATH)
    print(f"Flat index: {index_flat.ntotal} vectors, dim {dim}")

    # --- IVF-PQ approximate index, for latency-at-scale story ---
    nlist = max(8, int(np.sqrt(n)))          # rule of thumb: ~sqrt(N) coarse cells
    m = 8 if dim % 8 == 0 else 4             # PQ sub-quantizers must divide dim
    while dim % m != 0:
        m -= 1
    quantizer = faiss.IndexFlatIP(dim)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    index_ivfpq.train(embs)
    index_ivfpq.add(embs)
    index_ivfpq.nprobe = min(16, nlist)
    faiss.write_index(index_ivfpq, INDEX_IVFPQ_PATH)
    print(f"IVF-PQ index: nlist={nlist}, m={m}, nprobe={index_ivfpq.nprobe}")


if __name__ == "__main__":
    main()
