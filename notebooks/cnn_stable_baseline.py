import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.training.dataset_preparation import build_tf_dataset
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, MODEL_ROOT
from src.functions import train_val_test_sets, set_seed, cnn_predict, evaluate_thresholds, cnn_steps
import math

set_seed(SEED)

TRAIN_SPLIT = OUTPUT_NPY / "train_split.csv"
VAL_SPLIT = OUTPUT_NPY / "val_split.csv"
TEST_SPLIT = OUTPUT_NPY / "test_split.csv"

train_ds, val_ds, test_ds = train_val_test_sets(
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, BATCH_SIZE, SEED
)

train_steps, val_steps, test_steps = cnn_steps(
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)

def build_baseline_cnn(input_shape=(224, 224, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(32, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(128, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(256, 3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model

model = build_baseline_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall_40", thresholds=0.40),
        tf.keras.metrics.Precision(name="precision_40", thresholds=0.40),
        tf.keras.metrics.Recall(name="recall_50", thresholds=0.50),
        tf.keras.metrics.Precision(name="precision_50", thresholds=0.50),
    ]
)

model.summary()

print("EXPERIMENT = ADAM")
print(type(model.optimizer).__name__)
print(model.optimizer.get_config())

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

CHECKPOINT_PATH = MODEL_ROOT / "model_4blocks_adam.keras"
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    str(CHECKPOINT_PATH),
    monitor="val_auc",
    mode="max",
    save_best_only=True
)

history = model.fit(
    train_ds.repeat(),
    validation_data=val_ds.repeat(),
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[early_stop, checkpoint]
)

print("Best val_auc:", max(history.history["val_auc"]))
print("Epochs run:", len(history.history["val_auc"]))

model = tf.keras.models.load_model(str(CHECKPOINT_PATH))

y_prob, y_true = cnn_predict(model, test_ds)
evaluate_thresholds(y_prob, y_true)