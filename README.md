## Project Directory layout
```
scripts/
  gen_catalogue.py       synthetic catalogue generator (sandbox stand-in for scraping)
  scraper_template.py    the REAL scraper — run this on your own machine with internet
  build_index.py         embeds the catalogue, builds FAISS flat + IVF-PQ indices
backend/
  embedder.py             embedding function (classical CV; documented swap point for CLIP)
  matcher.py               retrieval + open-set accept/reject logic
  api.py                   FastAPI app (serves the API and the frontend)
  catalogue_meta.py
frontend/
  index.html               upload UI, calls the API, same-origin
eval/
  perturb.py               the 9 hard-condition image transforms
  gen_stumper_set.py       automated stumper set builder (Part B + "automate your stumper")
  gen_noncatalogue.py      non-catalogue distractor generator (open-set testing)
  calibrate_threshold.py   FAR/FRR threshold sweep
  run_eval.py              Part B evaluation harness (accuracy by condition)
  README.md                schema for plugging in real hand-shot phone photos
  report_auto.json         full results from the automated stumper run
  calibrated_thresholds.json
REPORT.md                  full write-up: methodology, results, honest failure analysis
```

## Running it
```bash
pip install faiss-cpu pillow numpy scikit-image fastapi "uvicorn[standard]" python-multipart

# 1. build the catalogue (or run scripts/scraper_template.py for a real one)
python3 scripts/gen_catalogue.py

# 2. embed it and build the indices
python3 scripts/build_index.py

# 3. (optional) rebuild the eval sets / recalibrate thresholds
python3 eval/gen_stumper_set.py
python3 eval/gen_noncatalogue.py
python3 eval/calibrate_threshold.py

# 4. run the Part B evaluation harness
python3 eval/run_eval.py --set auto

# 5. start the API + frontend
uvicorn backend.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/ in a browser
```

