# Example tuning files

Synthetic files showing the shapes the tuning tools read and write.
Nothing here comes from a real store; the subject is made up.

| File | Shape of |
|---|---|
| `replay.example.json` | a `topicsleuth tune synthesize` report: run metadata, the settings used, a summary with per-call diagnostics, and one entry per decision |
| `pairs.example.json` | a relationship label set: item, candidate topic, and a four-way label per pair |
| `model_key.example.json` | the blind key for a pair set: per pair id, the stratum and what the first pass decided; kept separate so labeling is blind |
| `labels.example.json` | a rater file: `{"rater", "labels": {pair_id: {"label", ...}}}`, the shape `scripts/compare_labels.py` compares |

Your own replays land in `tuning_runs/`, which is gitignored, as are
`data/labels/` and `data/tuning_baselines/`: a store's baselines and label
set describe the store's content and belong with the feeder that owns it,
not in this repository.

```bash
python scripts/diff_tuning_runs.py data/examples/replay.example.json tuning_runs/<new>.json
python scripts/compare_labels.py --key data/examples/model_key.example.json data/examples/labels.example.json
python scripts/label_pairs_openai.py --pairs data/examples/pairs.example.json --limit 1
```
