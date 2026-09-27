"""
Single source of truth for filesystem paths, computed relative to this
file's own location rather than hardcoded to any one machine's path.
Every other module imports from here instead of hardcoding
'/home/claude/matcher' — that path only ever existed in the sandbox this
project was originally built in, and does not exist on your machine.
"""
import os

# backend/ is one level below the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAT_DIR = os.path.join(PROJECT_ROOT, "catalogue")
IMG_DIR = os.path.join(CAT_DIR, "images")
META_PATH = os.path.join(CAT_DIR, "metadata.json")
IDS_PATH = os.path.join(CAT_DIR, "ids.json")
EMBEDDINGS_PATH = os.path.join(CAT_DIR, "embeddings.npy")
INDEX_FLAT_PATH = os.path.join(CAT_DIR, "index_flat.faiss")
INDEX_IVFPQ_PATH = os.path.join(CAT_DIR, "index_ivfpq.faiss")

EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
STUMPER_AUTO_DIR = os.path.join(EVAL_DIR, "stumper_auto")
STUMPER_AUTO_LABELS = os.path.join(EVAL_DIR, "stumper_auto_labels.json")
NONCAT_DIR = os.path.join(EVAL_DIR, "noncatalogue")
CALIBRATED_THRESHOLDS_PATH = os.path.join(EVAL_DIR, "calibrated_thresholds.json")
HANDSHOT_DIR = os.path.join(EVAL_DIR, "handshot")
HANDSHOT_LABELS = os.path.join(EVAL_DIR, "handshot_labels.json")

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
