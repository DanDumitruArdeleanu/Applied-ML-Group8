import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# This script evaluates trial scores from a JSON file and generates visualizations
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

# Define the root directory for evaluation
ROOT_DIR = os.path.join(PROJECT_ROOT, 'evaluation')
os.makedirs(ROOT_DIR, exist_ok=True)

# Define paths for trial scores and output directory
SCORES = os.path.join(PROJECT_ROOT, 'hyperparameter_optimisation', 'exported_dense_normal_model', 'trial_scores.json')
OUT_DIR = os.path.join(ROOT_DIR, 'visual_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

with open(SCORES) as f:
    trial_scores = json.load(f)

# Sort trial scores
scores = [(tid, score) for tid, score in trial_scores.items()]
scores.sort(key=lambda x: x[1], reverse=True)

# Extract trial IDs and scores
trial_ids = [tid for tid, _ in scores]
score_values = [score for _, score in scores]

# Summary statistics
best = max(score_values)
worst = min(score_values)
mean = np.mean(score_values)
std = np.std(score_values)
best_trial = scores[0][0]
worst_trial = scores[-1][0]

sem = stats.sem(score_values)  # Standard Error of the Mean
ci_low, ci_high = stats.t.interval(0.95, len(score_values)-1, loc=mean, scale=sem)  # 95% Confidence Interval

# Check if baseline falls within the confidence interval
statistically_significant = not (ci_low <= worst <= ci_high)

SUMMARY = {
    "num_trials": len(score_values),
    "best_trial_id": best_trial,
    "worst_trial_id": worst_trial,
    "best_performance": best,
    "worst_performance (baseline)": worst,
    "mean_performance": mean,
    "std_performance": std,
    "standard_error_of_the_mean": sem,
    "95%_confidence_interval": [round(ci_low, 6), round(ci_high, 6)],
    "statistically_significant_vs_baseline": statistically_significant
}

# Save summary statistics to a JSON file
SUMMARY_FILE = os.path.join(ROOT_DIR, 'numerical_analysis', "score_analysis.json")
with open(SUMMARY_FILE, 'w') as f:
    json.dump(SUMMARY, f, indent=4)

# Bar plot
plt.figure(figsize=(max(20, len(trial_ids) * 0.5), 6))
plt.bar(trial_ids, score_values, color='#4682B4', edgecolor='black')
plt.xticks(rotation=90)
plt.title("Validation Cosine Similarity per Trial")
plt.xlabel("Trial ID")
plt.ylabel("Cosine Similarity")
plt.grid(axis='y')
plt.xlim(-1, len(trial_ids))
plt.ylim(0.5, best + 0.01)
plt.tight_layout()
BAR_PATH = os.path.join(OUT_DIR, "score_barplot.png")
plt.savefig(BAR_PATH)
plt.close()

# Top N trials
TOP_N = 5
top_scores = scores[:TOP_N]
top_ids = [tid for tid, _ in top_scores]
top_values = [score for _, score in top_scores]

# Bar plot for top N trials
plt.figure(figsize=(TOP_N * 1.5, 6))
plt.bar(top_ids, top_values, color='#4682B4', edgecolor='black')
plt.xticks(rotation=45)
plt.title(f"Top {TOP_N} Performing Trials")
plt.xlabel("Trial ID")
plt.ylabel("Cosine Similarity")
plt.grid(axis='y')
plt.tight_layout(pad=3)
plt.ylim(min(top_values) - 0.022, max(top_values) + 0.002)
TOP_PATH = os.path.join(OUT_DIR, f"score_top_{TOP_N}_barplot.png")
plt.savefig(TOP_PATH)
plt.close()
