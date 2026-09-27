"""
FastAPI backend. Run with (from the project root, wherever you cloned it):
  uvicorn backend.api:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health
  POST /match          multipart file upload -> top-k candidates + confidence
  GET  /catalogue/{item_id}/image   serves a catalogue image (for the frontend thumbnails)
  GET  /stats           catalogue size, index type, calibrated thresholds
"""
import os, io, json, time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .matcher import Matcher
from .catalogue_meta import load_metadata_by_id
from .paths import IMG_DIR, IDS_PATH, INDEX_FLAT_PATH, CALIBRATED_THRESHOLDS_PATH, FRONTEND_DIR

app = FastAPI(title="Catalogue Matcher API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_meta = load_metadata_by_id()

# load calibrated thresholds if available, else fall back to matcher defaults
if os.path.exists(CALIBRATED_THRESHOLDS_PATH):
    with open(CALIBRATED_THRESHOLDS_PATH) as f:
        _calib = json.load(f)
    _sim_t, _margin_t = _calib["sim_threshold"], _calib["margin_threshold"]
else:
    _sim_t, _margin_t = 0.45, 0.0

matcher = Matcher(
    INDEX_FLAT_PATH,
    IDS_PATH,
    _meta,
    sim_threshold=_sim_t,
    margin_threshold=_margin_t,
)


@app.get("/health")
def health():
    return {"status": "ok", "catalogue_size": len(_meta)}


@app.get("/stats")
def stats():
    return {
        "catalogue_size": len(_meta),
        "index_type": "faiss IndexFlatIP (exact cosine)",
        "embedding": "HSV color histogram (4x4 spatial grid) + HOG shape descriptor, "
                     "classical CV — see backend/embedder.py docstring for why, "
                     "and how to swap in CLIP",
        "sim_threshold": _sim_t,
        "margin_threshold": _margin_t,
    }


@app.post("/match")
async def match(file: UploadFile = File(...), k: int = 5, autocrop: bool = False):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not read image file")

    t0 = time.perf_counter()
    result = matcher.match(img, k=k, autocrop=autocrop)
    total_ms = (time.perf_counter() - t0) * 1000

    response = {
        "is_match": result["is_match"],
        "reason": result["reason"],
        "top1_similarity": round(result["top1_similarity"], 4),
        "margin": round(result["margin"], 4),
        "timing_ms": result["timing_ms"],
        "request_total_ms": round(total_ms, 2),
        "candidates": [
            {
                "item_id": c["item_id"],
                "similarity": round(c["similarity"], 4),
                "confidence_pct": round(max(0.0, min(1.0, c["similarity"])) * 100, 1),
                "meta": c["meta"],
                "image_url": f"/catalogue/{c['item_id']}/image",
            }
            for c in result["candidates"]
        ],
    }
    return JSONResponse(response)


@app.get("/catalogue/{item_id}/image")
def catalogue_image(item_id: str):
    meta = _meta.get(item_id)
    if not meta:
        raise HTTPException(404, "unknown item_id")
    path = os.path.join(IMG_DIR, meta["file"])
    return FileResponse(path, media_type="image/jpeg")


# Serve the frontend at "/" — mounted last so it doesn't shadow the API routes above.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
