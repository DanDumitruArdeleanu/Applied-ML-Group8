# scripts/config.py
import os

# scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# project root (script_directory)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocessed_data")
POSTPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "postprocessed_data")

# postprocessed subfolders
SCREENSHOTS_DIR = os.path.join(POSTPROCESSED_DATA_DIR, "screenshots")
NORMALS_DIR = os.path.join(POSTPROCESSED_DATA_DIR, "normals")

# predicted data outputs
PREDICTED_DATA_DIR = os.path.join(DATA_DIR, "predicted_data")

# hyperparameter optimisation root and outputs
HYPEROPT_ROOT_DIR = os.path.join(BASE_DIR, "hyperparameter_optimisation")
HYPEROPT_EXPORT_DIR = os.path.join(HYPEROPT_ROOT_DIR, "exported_dense_normal_model")

# kt-dense-norm tuning
KT_DENSE_NORM_PRED_DIR = os.path.join(
    HYPEROPT_ROOT_DIR,
    "kt_dense_norm_tuning",
    "dense_norm_pred"
)

# object-level subdirectories
OBJ_NAMES = [f"obj_{i:02d}" for i in range(1, 15)]
SCREENSHOTS_OBJ_DIRS = {obj: os.path.join(SCREENSHOTS_DIR, obj) for obj in OBJ_NAMES}
NORMALS_OBJ_DIRS     = {obj: os.path.join(NORMALS_DIR, obj) for obj in OBJ_NAMES}
PREDICTED_DATA_OBJ_DIRS = {obj: os.path.join(PREDICTED_DATA_DIR, obj) for obj in OBJ_NAMES}
