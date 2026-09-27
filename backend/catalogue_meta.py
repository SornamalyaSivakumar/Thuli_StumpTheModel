import json
from .paths import META_PATH


def load_metadata_by_id():
    with open(META_PATH) as f:
        items = json.load(f)
    return {item["item_id"]: item for item in items}
