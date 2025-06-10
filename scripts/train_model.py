import os
import glob
import json
import numpy as np
import cv2
import tensorflow as tf
import keras_tuner as kt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow import keras
from config import (
    SCREENSHOTS_OBJ_DIRS,
    NORMALS_OBJ_DIRS,
    HYPEROPT_ROOT_DIR,
    HYPEROPT_EXPORT_DIR,
    PREDICTED_DATA_OBJ_DIRS,
    OBJ_NAMES,
)

# Ensure output directories exist
# Tuner output under hyperparameter_optimisation/kt_dense_norm_tuning/dense_norm_pred
TUNER_PARENT_DIR = os.path.join(HYPEROPT_ROOT_DIR, 'kt_dense_norm_tuning')
os.makedirs(TUNER_PARENT_DIR, exist_ok=True)
# Exported models and hyperparams directory
os.makedirs(HYPEROPT_EXPORT_DIR, exist_ok=True)
# Predicted validation normal maps per object
for obj in OBJ_NAMES:
    os.makedirs(PREDICTED_DATA_OBJ_DIRS[obj], exist_ok=True)

# Constants
TRAIN_SIZE = (128, 128)

# Custom tuner to optimize batch size
class BatchSizeHyperband(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.get('batch_size')
        return super().run_trial(trial, *args, **kwargs)

# Load screenshot input and normal map target data
def load_data(target=TRAIN_SIZE):
    X, y, paths = [], [], []
    for obj in OBJ_NAMES:
        ssf = SCREENSHOTS_OBJ_DIRS[obj]
        nmf = NORMALS_OBJ_DIRS[obj]
        for ss_path in sorted(glob.glob(os.path.join(ssf, 'screenshot_*.png'))):
            fid = os.path.basename(ss_path).replace('screenshot_', '')
            nm_path = os.path.join(nmf, f'normalmap_{fid}')
            if not os.path.exists(nm_path):
                continue
            img = cv2.imread(ss_path)
            if img is None:
                continue
            img = cv2.resize(img, target)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            nm_img = cv2.imread(nm_path, cv2.IMREAD_UNCHANGED)
            if nm_img is None:
                continue
            nm_img = cv2.resize(nm_img, target)
            nm_img = cv2.cvtColor(nm_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            nm_norm = nm_img * 2.0 - 1.0
            row_norms = np.linalg.norm(nm_norm, axis=-1, keepdims=True) + 1e-9
            nm_norm = nm_norm / row_norms
            X.append(img)
            y.append(nm_norm)
            paths.append(ss_path)
    return np.array(X), np.array(y), paths

# Masked cosine loss
def masked_cosine_loss(y_true, y_pred):
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1) > 1e-5
    mask = tf.cast(mask, y_pred.dtype)
    y_true_norm = tf.math.l2_normalize(y_true, axis=-1)
    cos = tf.reduce_sum(y_true_norm * y_pred, axis=-1)
    loss = (1.0 - cos) * mask
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

# Model builder for Hyperparameter tuning
def build_model(hp):
    blocks = hp.Int('blocks', 2, 4, default=3)
    bot_filters = hp.Int('filters_bottleneck', 64, 256, step=64, default=128)
    lr = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log', default=1e-4)
    hp.Choice('batch_size', [4, 8, 16, 32], default=8)
    l2 = hp.Float('l2_reg', 1e-6, 1e-2, sampling='log', default=1e-4)
    drop = hp.Float('dropout', 0.0, 0.5, step=0.1, default=0.1)

    inputs = keras.Input((*TRAIN_SIZE, 3), name='input_image')
    x, skips = inputs, []
    for i in range(blocks):
        f = hp.Int(f'filters_{i}', 32, 128, step=32, default=64)
        ks = hp.Choice(f'kernel_size_{i}', [3, 5], default=3)
        x = keras.layers.Conv2D(f, ks, padding='same', kernel_regularizer=keras.regularizers.l2(l2))(x)
        x = keras.layers.Activation('gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(drop)(x)
        skips.append(x)
        x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(bot_filters, 3, padding='same', kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = keras.layers.Activation('gelu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(drop)(x)
    for i, skip in enumerate(reversed(skips)):
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Concatenate()([x, skip])
        uf = hp.Int(f'filters_up_{i}', 32, 128, step=32, default=64)
        x = keras.layers.Conv2D(uf, 3, padding='same', kernel_regularizer=keras.regularizers.l2(l2))(x)
        x = keras.layers.Activation('gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(drop)(x)
    outputs = keras.layers.Conv2D(3, 1, padding='same')(x)
    outputs = keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name='output_normals')(outputs)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=masked_cosine_loss,
                  metrics=[keras.metrics.CosineSimilarity(axis=-1, name='cos_sim')])
    return model

# Main execution
if __name__ == '__main__':
    # Load data
    X, y, paths = load_data()
    if X.size == 0:
        print('No data found in specified obj_01–obj_14 subfolders. Exiting.')
        exit(0)
    X_train, X_val, y_train, y_val, p_train, p_val = train_test_split(
        X, y, paths, test_size=0.2, random_state=42
    )

    # Hyperparameter tuning
    tuner = BatchSizeHyperband(
        build_model,
        objective=kt.Objective('val_cos_sim', direction='max'),
        max_epochs=20,
        factor=3,
        directory=TUNER_PARENT_DIR,
        project_name='dense_norm_pred',
        overwrite=False
    )
    callbacks = [
        keras.callbacks.EarlyStopping('val_cos_sim', patience=5, mode='max', restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

    # Retrieve and filter completed trials
    trials = tuner.oracle.trials
    scored = [(tid, t.score) for tid, t in trials.items() if t.score is not None]
    best_id, best_score = max(scored, key=lambda x: x[1])
    good = [item for item in scored if item[1] > 0]
    worst_id, worst_score = (min(good, key=lambda x: x[1]) if good else (None, None))

    # Load best and worst models
    best_model = tuner.get_best_models(num_models=1)[0]
    worst_model = None
    if worst_id:
        worst_model = tuner.get_best_models(num_models=2)[1]

    # Save exported models and hyperparameters
    best_model.save(os.path.join(HYPEROPT_EXPORT_DIR, 'best_dense_normal_model.keras'), include_optimizer=True)
    with open(os.path.join(HYPEROPT_EXPORT_DIR, 'best_hyperparameters.json'), 'w') as bf:
        json.dump(trials[best_id].hyperparameters.values, bf, indent=4)
    if worst_model:
        worst_model.save(os.path.join(HYPEROPT_EXPORT_DIR, 'worst_dense_normal_model.keras'), include_optimizer=True)
        with open(os.path.join(HYPEROPT_EXPORT_DIR, 'worst_hyperparameters.json'), 'w') as wf:
            json.dump(trials[worst_id].hyperparameters.values, wf, indent=4)

    # Performance summary
    perf = {
        'best_performance': float(best_score),
        'worst_performance': float(worst_score) if worst_score is not None else None
    }
    with open(os.path.join(HYPEROPT_EXPORT_DIR, 'performance_summary.json'), 'w') as pf:
        json.dump(perf, pf, indent=4)

    # Generate and save validation predictions per object
    def save_predictions(model, label):
        preds = model.predict(X_val)
        for ss_path, pred in zip(p_val, preds):
            obj = Path(ss_path).parent.name
            odir = os.path.join(PREDICTED_DATA_OBJ_DIRS[obj], label)
            os.makedirs(odir, exist_ok=True)
            rgb = ((np.clip((pred + 1) / 2, 0, 1) * 255).astype(np.uint8))
            cv2.imwrite(
                os.path.join(odir, os.path.basename(ss_path).replace('screenshot_', '')),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )

    save_predictions(best_model, 'best_prediction')
    if worst_model:
        save_predictions(worst_model, 'worst_prediction')

    print('Tuning, export, and validation predictions complete.')