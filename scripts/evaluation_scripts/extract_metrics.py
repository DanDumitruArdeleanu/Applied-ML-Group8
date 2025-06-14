import os
import json

# This script extracts the final training and validation loss and accuracy from the trials
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

# Define paths for trials and output files
TRIALS_BASE = os.path.join(PROJECT_ROOT, 'hyperparameter_optimisation', 'kt_dense_norm_tuning', 'dense_norm_pred')
ORACLE = os.path.join(TRIALS_BASE, 'oracle.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'evaluation', 'numerical_analysis')

os.makedirs(OUT_DIR, exist_ok=True)

# Output file paths
TRAIN_LOSS_FILE = os.path.join(OUT_DIR, 'final_train_loss.json')
VAL_LOSS_FILE = os.path.join(OUT_DIR, 'final_val_loss.json')
TRAIN_ACC_FILE = os.path.join(OUT_DIR, 'final_train_accuracy.json')
VAL_ACC_FILE = os.path.join(OUT_DIR, 'final_val_accuracy.json')

# Load trial IDs
ids = []
if os.path.isfile(ORACLE):
    o = json.load(open(ORACLE))
    ids = o.get('start_order') \
       or list(o.get('id_to_hash', {}).keys())

# If no IDs are found in the oracle, fallback to directory listing
if not ids:
    ids = [d.replace('trial_', '') for d in os.listdir(TRIALS_BASE)
           if d.startswith('trial_') and d.split('_', 1)[1].isdigit()]

train_loss, val_loss, train_acc, val_acc = {}, {}, {}, {}

# Extract metrics from each trial
for tid in ids:
    dname = f"trial_{tid}"
    path = os.path.join(TRIALS_BASE, dname, 'trial.json')

    if not os.path.isfile(path):
        continue

    trial = json.load(open(path))

    # Extracting metrics from the trial
    metrics = trial.get('metrics', {}).get('metrics', {})
    train_loss_obs = metrics.get('loss', {}).get('observations', [])
    val_loss_obs = metrics.get('val_loss', {}).get('observations', [])
    train_acc_obs = metrics.get('cos_sim', {}).get('observations', [])
    val_acc_obs = metrics.get('val_cos_sim', {}).get('observations', [])

    if train_loss_obs:
        train_loss[tid] = train_loss_obs[-1]['value'][0]
    if val_loss_obs:
        val_loss[tid] = val_loss_obs[-1]['value'][0]
    if train_acc_obs:
        train_acc[tid] = train_acc_obs[-1]['value'][0]
    if val_acc_obs:
        val_acc[tid] = val_acc_obs[-1]['value'][0]

OUTPUTS = {
    TRAIN_LOSS_FILE: train_loss,
    VAL_LOSS_FILE: val_loss,
    TRAIN_ACC_FILE: train_acc,
    VAL_ACC_FILE: val_acc
}

# Write the extracted metrics to JSON files
for file, data in OUTPUTS.items():
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)
