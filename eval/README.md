# Hand-shot stumper set — how to plug in your real photos

This sandbox has no camera, so `eval/stumper_auto*` (programmatically
perturbed catalogue images) stands in for the required "100 real phone
photos" deliverable. When you run this for real:

## 1. Shoot the photos
100+ photos of items genuinely in your catalogue, on a phone, deliberately
hard: bad lighting, odd angles, partial occlusion, cluttered background,
motion blur, reflections, a hand/wrist in frame. Save them into
`eval/handshot/` as `img_001.jpg`, `img_002.jpg`, etc.

Shoot each condition as its own deliberate attempt, not incidentally —
you want clean coverage of each failure mode, plus some multi-condition
shots (e.g. bad lighting AND cluttered background at once), the same way
`eval/gen_stumper_set.py` stacks two synthetic conditions.

## 2. Label them
Create `eval/handshot_labels.json` as a list of objects, one per photo:

```json
[
  {
    "file": "img_001.jpg",
    "true_item_id": "SKU003421",
    "conditions": ["low_light", "cluttered_background"]
  },
  {
    "file": "img_002.jpg",
    "true_item_id": "SKU000117",
    "conditions": ["motion_blur"]
  }
]
```

Use the same condition vocabulary as `eval/perturb.py`'s `CONDITIONS` list
(`low_light`, `overexposed`, `motion_blur`, `rotation`, `partial_occlusion`,
`cluttered_background`, `reflection_glare`, `off_angle_crop`, `low_res`) —
or add your own (e.g. `hand_in_frame`, `wrist_shot`) if a phone shoot
surfaces a failure mode the synthetic generator can't produce. Record
`true_item_id` from your own catalogue metadata at shoot time, not from
guessing afterward — that's what makes it ground truth.

## 3. Run the harness
```
python3 eval/run_eval.py --set handshot
```
This produces the identical report structure (overall + per-condition +
per-stack-depth accuracy, worst failures by confidence) as the automated
set, saved to `eval/report_handshot.json`, so the two are directly
comparable — which is the whole point of the "automate your stumper"
extension: showing whether your generator produces harder or easier
cases than a human deliberately trying to defeat the matcher.
