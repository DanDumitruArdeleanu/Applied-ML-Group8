# Surface Normals Prediction

This repository implements a full pipeline for predicting surface normal maps from object screenshots, including data preprocessing, model training with hyperparameter tuning, and a FastAPI for inference.

---

Running the API Server

1. **Start the server** (from project root):

   ```bash
   uvicorn scripts.main:app --reload
   ```

2. **Browse the API docs**:
   Open your browser to:

   ```
   http://127.0.0.1:8000/docs
   ```

   Use the interactive docs to test the `/predict/` endpoint.
