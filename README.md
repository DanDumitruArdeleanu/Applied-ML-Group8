# Surface Normals Prediction

This repository implements a full pipeline for predicting surface normal maps from object screenshots, including data preprocessing, model training with hyperparameter tuning, and a FastAPI for inference.

---

## 📁 Directory Hierarchy

```
script_directory/  (project root)
├─ data/
│  ├─ raw_data/                                 # Original .ply mesh files: obj_01.ply … obj_14.ply
│  ├─ preprocessed_data/                        # Feature-extraction & analysis outputs
│  ├─ postprocessed_data/                       # Rendered screenshots & normals
│  │  ├─ screenshots/
│  │  │  ├─ obj_01/ … obj_14/                   # RGB-lit screenshots per orientation
│  │  └─ normals/
│  │     ├─ obj_01/ … obj_14/                   # Normal-map renders per orientation
│  └─ predicted_data/                           # Model validation predictions
│     ├─ obj_01/ … obj_14/                      # Predictions per object (best & worst)
├─ hyperparameter_optimisation/_optimisation/
│  ├─ kt_dense_norm_tuning/
│  │  └─ dense_norm_pred/                       # Keras-Tuner outputs & trials
│  │     ├─ trial_0000/ … trial_0029
│  │     └─ tuner0.json                         # Tuner state for reload
│  └─ exported_dense_normal_model/              # Final models, hyperparameters, & summaries
│     ├─ best_dense_normal_model.keras
│     ├─ best_hyperparameters.json
│     ├─ worst_dense_normal_model.keras
│     ├─ worst_hyperparameters.json
│     ├─ performance_summary.json
│     └─ trial_scores.json                      # Aggregated trial scores
├─ scripts/                                     # All Python scripts
│  ├─ config.py                                 # Centralized path config
│  ├─ data_datapreprocessing.py                 # Raw → PCA features
│  ├─ data_postprocessing.py                    # Rotate & render meshes per orientation
│  ├─ extract_models.py                         # Summarize trial scores into JSON
│  ├─ train_model.py                            # Hyperparameter tuning & model export
│  ├─ main.py                                   # FastAPI server entrypoint
├─ requirements.txt                             # Third-party dependencies
└─ README.md                                    # You are here
```

---

## Running The Project With Docker

1. **Clone the repo** and navigate into the project directory:

   ```bash
   git clone https://github.com/DanDumitruArdeleanu/Applied-ML-Group8.git
   cd Applied-ML-Group8
   ```
2. **Download Docker Desktop**, from the following link:
   ```
   https://www.docker.com/products/docker-desktop/
   ```

3. **Build The Docker Image**:
   ```
   docker build --shm-size=2g -t my-ml-app .
   ```

4. **Run Docker**:
   ```
   docker run --shm-size=2g -p 8000:8000 my-ml-app
   ```

