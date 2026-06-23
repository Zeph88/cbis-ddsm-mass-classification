import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.training.cnn_evaluation import cnn_predict, evaluate_thresholds
from src.training.dataset_preparation import cnn_steps, train_val_test_sets, build_tf_dataset
from src.functions import set_seed
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_MODEL, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, PIXELS_H, PIXELS_W
import math

set_seed(SEED)

zoom_to_roi=False
resolution=(PIXELS_H, PIXELS_W)


if zoom_to_roi:
    file_path = f"zoom_{resolution[0]}x{resolution[1]}_test"
else:
    file_path = f"full_{resolution[0]}x{resolution[1]}_test"

print(f"dataset_index_{file_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{file_path}.csv")

train_ds = build_tf_dataset(
        df[df["set"]=="train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

test_ds = build_tf_dataset(
        df[df["set"]=="test"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

train_size = len(train_ds)
train_steps = math.ceil(train_size / BATCH_SIZE)

def build_baseline_cnn(input_shape=(598, 598, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    for filters in [16, 32, 64]:
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return tf.keras.Model(inputs, outputs)

model = build_baseline_cnn(input_shape=(PIXELS_H, PIXELS_W, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Recall(name="recall_35", thresholds=0.35),
        tf.keras.metrics.Precision(name="precision_35", thresholds=0.35),
        tf.keras.metrics.Recall(name="recall_40", thresholds=0.40),
        tf.keras.metrics.Precision(name="precision_40", thresholds=0.40),
        tf.keras.metrics.Recall(name="recall_45", thresholds=0.45),
        tf.keras.metrics.Precision(name="precision_45", thresholds=0.45),
        tf.keras.metrics.Recall(name="recall_50", thresholds=0.50),
        tf.keras.metrics.Precision(name="precision_50", thresholds=0.50),
    ]
)

model.summary()


for x, y in train_ds.take(1):
    print(x.shape)
    print(y.shape)

history = model.fit(
    train_ds,
    epochs=EPOCHS
)

model = tf.keras.models.load_model(OUTPUT_MODEL / "lightweigh_model_test.keras")

y_prob, y_true = cnn_predict(model, test_ds)
evaluate_thresholds(y_prob, y_true)