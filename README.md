# Surface Normals Prediction

This repository implements a full pipeline for predicting surface normal maps from object screenshots, including data preprocessing, model training with hyperparameter tuning, and a FastAPI for inference.

---

## 🚀 Setup & Installation

1. **Clone the repo** and change into the project directory:

   ```bash
   git clone https://github.com/DanDumitruArdeleanu/Applied-ML-Group8.git
   cd Applied-ML-Group8
   ```

2. **Create a virtual environment** and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   # .\.venv\Scripts\activate   # Windows PowerShell
   ```

3. **Install required packages**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


### 2. Running the API Server

1. **Start the server** (from project root):

   ```bash
   cd Applied-ML-Group8/fast_api
   uvicorn app.main:app --reload
   ```

2. **Browse the API docs**:
   Open your browser to:

   ```
   http://127.0.0.1:8000/docs
   ```

   Use the interactive docs to test the `/predict/` endpoint.
