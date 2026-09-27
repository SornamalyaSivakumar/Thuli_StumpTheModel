"""
Part B evaluation harness.

Runs the matcher over a labeled stumper set (list of {file, true_item_id,
conditions[]}) and reports:
  - overall top-1 / top-5 accuracy
  - accuracy broken down by EACH individual condition (a stacked example
    with conditions=[a,b] counts toward both a's and b's bucket, since we
    want to know "how much does motion_blur hurt", not just "how do the
    exact combos perform")
  - accuracy broken down by NUMBER of stacked conditions (0/1/2) as a
    blunter but useful difficulty signal
  - confusion notes: for wrong answers, what was predicted instead and
    at what confidence, so failures are diagnosable rather than just a
    percentage

Works against either the automated stumper set (eval/stumper_auto*) or a
hand-shot set that follows the same {file, true_item_id, conditions[]}
schema (see eval/README.md for the schema your phone-shot labels.csv
should be converted to).
"""
import sys, os, json, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from backend.matcher import Matcher
from backend.catalogue_meta import load_metadata_by_id
from backend.paths import (CAT_DIR, IDS_PATH, EVAL_DIR, STUMPER_AUTO_DIR,
                            STUMPER_AUTO_LABELS, HANDSHOT_DIR, HANDSHOT_LABELS,
                            CALIBRATED_THRESHOLDS_PATH)


def run_eval(image_dir, labels_path, index_name="index_flat.faiss",
             sim_threshold=0.45, margin_threshold=0.0, top_k=5, autocrop=False):
    meta = load_metadata_by_id()
    m = Matcher(os.path.join(CAT_DIR, index_name), IDS_PATH,
                meta, sim_threshold=sim_threshold, margin_threshold=margin_threshold,
                top_k=top_k)

    with open(labels_path) as f:
        labels = json.load(f)

    per_condition = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})
    per_stack_len = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0})
    overall = {"n": 0, "top1": 0, "top5": 0, "rejected": 0}
    failures = []
    latencies = []

    for lab in labels:
        img = Image.open(os.path.join(image_dir, lab["file"]))
        result = m.match(img, autocrop=autocrop)
        latencies.append(result["timing_ms"]["total"])
        pred_ids = [c["item_id"] for c in result["candidates"]]
        true_id = lab["true_item_id"]
        top1_correct = len(pred_ids) > 0 and pred_ids[0] == true_id
        top5_correct = true_id in pred_ids

        overall["n"] += 1
        overall["top1"] += int(top1_correct)
        overall["top5"] += int(top5_correct)
        if not result["is_match"]:
            overall["rejected"] += 1

        conditions = lab.get("conditions", ["clean"])
        for c in conditions:
            per_condition[c]["n"] += 1
            per_condition[c]["top1"] += int(top1_correct)
            per_condition[c]["top5"] += int(top5_correct)

        stack_len = len([c for c in conditions if c != "clean"])
        per_stack_len[stack_len]["n"] += 1
        per_stack_len[stack_len]["top1"] += int(top1_correct)
        per_stack_len[stack_len]["top5"] += int(top5_correct)

        if not top1_correct:
            failures.append({
                "file": lab["file"], "true_item_id": true_id,
                "conditions": conditions,
                "predicted_top1": pred_ids[0] if pred_ids else None,
                "top1_similarity": result["top1_similarity"],
                "correct_rank": (pred_ids.index(true_id) + 1) if true_id in pred_ids else None,
                "was_rejected": not result["is_match"],
            })

    report = {
        "n_images": overall["n"],
        "top1_accuracy": overall["top1"] / overall["n"],
        "top5_accuracy": overall["top5"] / overall["n"],
        "reject_rate": overall["rejected"] / overall["n"],
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2],
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
        },
        "by_condition": {
            c: {"n": v["n"], "top1_accuracy": round(v["top1"] / v["n"], 3),
                "top5_accuracy": round(v["top5"] / v["n"], 3)}
            for c, v in sorted(per_condition.items(), key=lambda kv: kv[1]["top1"] / kv[1]["n"])
        },
        "by_num_stacked_conditions": {
            str(k): {"n": v["n"], "top1_accuracy": round(v["top1"] / v["n"], 3)}
            for k, v in sorted(per_stack_len.items())
        },
        "n_failures": len(failures),
    }
    return report, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["auto", "handshot"], default="auto")
    ap.add_argument("--index", default="index_flat.faiss")
    ap.add_argument("--autocrop", action="store_true")
    args = ap.parse_args()

    if args.set == "auto":
        image_dir = STUMPER_AUTO_DIR
        labels_path = STUMPER_AUTO_LABELS
    else:
        image_dir = HANDSHOT_DIR
        labels_path = HANDSHOT_LABELS
        if not os.path.exists(labels_path):
            print(f"No hand-shot labels found at {labels_path}.")
            print("This is expected in the sandbox demo — see eval/README.md "
                  "for the schema to fill in once you've shot real photos.")
            return

    report, failures = run_eval(image_dir, labels_path, index_name=args.index, autocrop=args.autocrop)

    print(json.dumps(report, indent=2))
    print(f"\n{len(failures)} failures. Worst 5 by similarity margin from correct:")
    for f in sorted(failures, key=lambda x: -x["top1_similarity"])[:5]:
        print(f"  {f['file']}: true={f['true_item_id']} conditions={f['conditions']} "
              f"predicted={f['predicted_top1']} sim={f['top1_similarity']:.3f} "
              f"true_item_rank={f['correct_rank']}")

    out_path = os.path.join(EVAL_DIR, f"report_{args.set}.json")
    with open(out_path, "w") as fo:
        json.dump({"report": report, "failures": failures}, fo, indent=2)
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
