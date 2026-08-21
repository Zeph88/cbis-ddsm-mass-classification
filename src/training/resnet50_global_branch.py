import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions, plot_training_metric, build_binary_metrics
from src.training.dataset_preparation import cnn_steps, train_val_test_sets
from src.functions import set_seed, ensure_directory
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_MODEL, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, GLOBAL_HEIGHT, GLOBAL_WIDTH, OUTPUT_PLOT
import math
import gc
from src.modeling.global_resnet50 import build_global_model


tf.keras.backend.clear_session()
set_seed(SEED)

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

resolution=(GLOBAL_HEIGHT, GLOBAL_WIDTH)

file_path = f"full_{resolution[0]}x{resolution[1]}"

print(f"dataset_index_{file_path}.csv")

df = pd.read_csv(OUTPUT_NPY / f"dataset_index_{file_path}.csv")

train_ds, val_ds, test_ds = train_val_test_sets(
    df, BATCH_SIZE, SEED
)

for images, labels in train_ds.take(1):
    print("Batch shape:", images.shape)
    print("Minimum:", tf.reduce_min(images).numpy())
    print("Maximum:", tf.reduce_max(images).numpy())
    print("Mean:", tf.reduce_mean(images).numpy())
    print("Standard deviation:", tf.math.reduce_std(images).numpy())
    print("Labels:", labels.numpy())

del images, labels
gc.collect()

train_steps, val_steps, test_steps = cnn_steps(df)

model = build_global_model(
    input_shape=(
        GLOBAL_HEIGHT,
        GLOBAL_WIDTH,
        1,
    ),
    seed=SEED,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=build_binary_metrics()
)

model.summary()

head_checkpoint_path = OUTPUT_MODEL / f"global_resnet50_head.keras"

callbacks_head = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        restore_best_weights=False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        head_checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.CSVLogger(
        OUTPUT_MODEL / f"global_resnet50_head.csv"
    ),
]
history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_head,
)

history_values = {
    metric_name: list(metric_values)
    for metric_name, metric_values in history_head.history.items()
}

del history_head
del callbacks_head
del model

del train_ds
del val_ds
del test_ds

tf.keras.backend.clear_session()
gc.collect()


print("\nHead-only model evaluation:")


model = tf.keras.models.load_model(
    head_checkpoint_path,
    compile=False,
)


EVAL_BATCH_SIZE = 2

eval_train_ds, eval_val_ds, test_ds_eval = train_val_test_sets(
    df,
    EVAL_BATCH_SIZE,
    SEED,
)

del eval_train_ds
del eval_val_ds
gc.collect()


y_true, y_prob = collect_binary_predictions(model, test_ds_eval)

for threshold in [0.35, 0.4, 0.45, 0.5]:
    metrics = calculate_metrics(y_true, y_prob, threshold)
    print(f"threshold : {threshold}, accuracy : {metrics["accuracy"]}, precision : {metrics["precision"]}, recall : {metrics["recall"]}")

del test_ds_eval
del model
tf.keras.backend.clear_session()
gc.collect()


# Loss plot
plot_training_metric(
    history=history_values,
    metric_name="loss",
    ylabel="Binary cross-entropy",
    title="Training and validation loss",
    output_path=(OUTPUT_PLOT / f"global head only - val_loss and loss - {GLOBAL_HEIGHT}x{GLOBAL_WIDTH} - seed {SEED}.png")
)

# AUC plot
plot_training_metric(
    history=history_values,
    metric_name="auc",
    ylabel="AUC",
    title="Training and validation AUC",
    output_path=(OUTPUT_PLOT / f"global head only - val_auc and auc - {GLOBAL_HEIGHT}x{GLOBAL_WIDTH} - seed {SEED}.png")
)