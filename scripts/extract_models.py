import json
from pathlib import Path
from config import KT_DENSE_NORM_PRED_DIR, HYPEROPT_EXPORT_DIR

# Base directory for trial results
base_dir = Path(KT_DENSE_NORM_PRED_DIR)
out_path = Path(HYPEROPT_EXPORT_DIR) / 'trial_scores.json'

scores = {}
# Iterate over each trial folder (trial_XXXX)
for trial_dir in sorted(base_dir.iterdir()):
    if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
        continue
    trial_file = trial_dir / 'trial.json'
    if not trial_file.is_file():
        print(f"Missing trial.json in {trial_dir.name}, skipping.")
        continue
    # Load each trial's JSON and extract overall score
    data = json.loads(trial_file.read_text())
    score = data.get('score')
    if score is not None:
        # trial ID without prefix
        tid = trial_dir.name.split('_', 1)[1]
        scores[tid] = score

# Ensure export directory exists
Path(HYPEROPT_EXPORT_DIR).mkdir(parents=True, exist_ok=True)
# Write aggregated scores
out_path.write_text(json.dumps(scores, indent=4))

print(f"Wrote {len(scores)}/{len(list(base_dir.glob('trial_*')))} scores to '{out_path}'")
