import os
import json
import numpy as np
import cv2
import tensorflow as tf
import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
from io import BytesIO
from fastapi.responses import StreamingResponse

keras.config.enable_unsafe_deserialization()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_dense_normal_model.keras")
HP_PATH = os.path.join(MODEL_DIR, "best_hyperparameters.json")
IMG_SIZE = (128, 128)


def masked_cosine_loss(y_true, y_pred):
    mask = tf.cast(tf.reduce_sum(tf.abs(y_true), axis=-1) > 1e-5, y_pred.dtype)
    cos = tf.reduce_sum(tf.math.l2_normalize(y_true, -1) * y_pred, -1)
    return tf.reduce_sum((1 - cos) * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def build_model(hp):
    inp = keras.Input((*IMG_SIZE, 3))
    x, skips = inp, []
    l2_reg = hp["l2_reg"]
    dropout_rate = hp["dropout"]

    for i in range(hp["blocks"]):
        x = keras.layers.Conv2D(
            hp[f"filters_{i}"],
            hp[f"kernel_size_{i}"],
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        skips.append(x)
        x = keras.layers.MaxPooling2D()(x)

    x = keras.layers.Conv2D(
        hp["filters_bottleneck"],
        3,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
    )(x)
    x = keras.layers.Activation("gelu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    for i, skip in enumerate(reversed(skips)):
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Concatenate()([x, skip])
        x = keras.layers.Conv2D(
            hp[f"filters_up_{i}"],
            3,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        )(x)
        x = keras.layers.Activation("gelu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    out = keras.layers.Conv2D(3, 1, padding="same")(x)
    out = keras.layers.Lambda(
        lambda t: tf.math.l2_normalize(t, -1), name="output_normals"
    )(out)
    return keras.Model(inp, out)


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not os.path.isfile(HP_PATH):
        raise RuntimeError(f"Missing hyperparameters: {HP_PATH}")
    with open(HP_PATH) as f:
        hp = json.load(f)

    model = build_model(hp)
    model.load_weights(MODEL_PATH)

    model.compile(
        optimizer="adam",
        loss=masked_cosine_loss,
        metrics=[keras.metrics.CosineSimilarity(axis=-1, name="cos_sim")],
    )
    yield

    model = None


app = FastAPI(title="Normal Map Prediction API", lifespan=lifespan)


def preprocess(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return np.expand_dims(img.astype(np.float32), 0)


def postprocess(pred: np.ndarray):
    nm = np.squeeze(pred, 0)
    rgb = ((np.clip((nm + 1) / 2, 0, 1)) * 255).astype(np.uint8)
    return rgb.tolist()


@app.get("/")
def root():
    return {"message": "API up and running"}


@app.post(
    "/predict/",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Predicted normal map as PNG image",
        }
    },
)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model not ready")
    try:
        data = await file.read()
        inp = preprocess(data)
        pred = model.predict(inp)
        nm = np.squeeze(pred, 0)
        rgb = ((np.clip((nm + 1) / 2, 0, 1)) * 255).astype(np.uint8)

        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not is_success:
            raise RuntimeError("Could not encode image")
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/png")
    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except Exception as e:
        raise HTTPException(500, f"Internal error: {e}")
