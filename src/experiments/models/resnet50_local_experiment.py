import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
)

from src.training.cnn_evaluation import (
    cnn_predict,
    evaluate_thresholds,
)
from src.training.dataset_preparation import (
    cnn_steps,
    train_val_test_sets,
)
from src.functions import set_seed, ensure_directory
from src.config import (
    OUTPUT_MODEL,
    OUTPUT_NPY,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    PIXELS_H,
    PIXELS_W,
    OUTPUT_PLOT,
)


# ================================================================
# Reproducibility
# ================================================================

set_seed(SEED)

ensure_directory(OUTPUT_MODEL)
ensure_directory(OUTPUT_PLOT)

# ================================================================
# Dataset preparation
# ================================================================

zoom_to_roi = True
resolution = (PIXELS_H, PIXELS_W)

if zoom_to_roi:
    file_path = f"zoom_{resolution[0]}x{resolution[1]}"
else:
    file_path = f"full_{resolution[0]}x{resolution[1]}"

dataset_index_path = (
    OUTPUT_NPY / f"dataset_index_{file_path}.csv"
)

print(f"Loading: {dataset_index_path}")

df = pd.read_csv(dataset_index_path)

train_ds, val_ds, test_ds = train_val_test_sets(
    df,
    BATCH_SIZE,
    SEED,
)

train_steps, val_steps, test_steps = cnn_steps(df)


# ================================================================
# Raw BCE metric
#
# The normal Keras loss includes the L2 regularisation penalty.
# This custom metric computes only the BCE of the predictions.
#
# It also fixes the rank mismatch:
#     y_true: (batch,)
#     y_pred: (batch, 1)
# ================================================================

@tf.keras.utils.register_keras_serializable(
    package="CustomMetrics"
)
def raw_bce(y_true, y_pred):
    y_true = tf.cast(
        y_true,
        dtype=y_pred.dtype,
    )

    y_true = tf.reshape(
        y_true,
        tf.shape(y_pred),
    )

    return tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )


# ================================================================
# External probability metrics
# ================================================================

def prepare_binary_arrays(y_true, y_prob):
    """
    Converts targets and predictions to flat NumPy arrays and clips
    probabilities to avoid log(0) in BCE calculations.
    """

    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_prob = np.asarray(y_prob).astype(np.float64).ravel()

    y_prob_clipped = np.clip(
        y_prob,
        1e-7,
        1.0 - 1e-7,
    )

    return y_true, y_prob, y_prob_clipped


def calculate_probability_metrics(
    y_true,
    y_prob,
    naive_probability,
):
    """
    Computes threshold-independent probability metrics.

    The naive model predicts the positive-class prevalence observed
    in the training set for every image.
    """

    y_true, y_prob, y_prob_clipped = prepare_binary_arrays(
        y_true,
        y_prob,
    )

    naive_probability = float(
        np.clip(
            naive_probability,
            1e-7,
            1.0 - 1e-7,
        )
    )

    naive_probabilities = np.full(
        shape=y_true.shape,
        fill_value=naive_probability,
        dtype=np.float64,
    )

    model_bce = log_loss(
        y_true,
        y_prob_clipped,
        labels=[0, 1],
    )

    naive_bce = log_loss(
        y_true,
        naive_probabilities,
        labels=[0, 1],
    )

    model_brier = brier_score_loss(
        y_true,
        y_prob,
    )

    naive_brier = brier_score_loss(
        y_true,
        naive_probabilities,
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    bce_gain = naive_bce - model_bce

    relative_bce_gain = (
        bce_gain / naive_bce
        if naive_bce > 0
        else np.nan
    )

    brier_gain = naive_brier - model_brier

    return {
        "auc": auc,
        "raw_bce": model_bce,
        "naive_bce": naive_bce,
        "bce_gain": bce_gain,
        "relative_bce_gain": relative_bce_gain,
        "brier_score": model_brier,
        "naive_brier_score": naive_brier,
        "brier_gain": brier_gain,
        "minimum_probability": float(y_prob.min()),
        "maximum_probability": float(y_prob.max()),
        "average_probability": float(y_prob.mean()),
        "positive_rate": float(y_true.mean()),
    }


def print_probability_report(
    dataset_name,
    metrics,
):
    """
    Prints the probability-quality report for one dataset.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name} probability metrics")
    print("=" * 70)

    print(
        "Observed positive rate: "
        f"{metrics['positive_rate']:.4f}"
    )

    print(
        "Minimum probability:    "
        f"{metrics['minimum_probability']:.8f}"
    )

    print(
        "Maximum probability:    "
        f"{metrics['maximum_probability']:.8f}"
    )

    print(
        "Average probability:    "
        f"{metrics['average_probability']:.4f}"
    )

    print(f"AUC:                    {metrics['auc']:.4f}")

    print(
        f"Raw BCE:                "
        f"{metrics['raw_bce']:.4f}"
    )

    print(
        f"Naive BCE:              "
        f"{metrics['naive_bce']:.4f}"
    )

    print(
        f"Absolute BCE gain:      "
        f"{metrics['bce_gain']:+.4f}"
    )

    print(
        f"Relative BCE gain:      "
        f"{metrics['relative_bce_gain']:+.2%}"
    )

    print(
        f"Brier score:            "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"Naive Brier score:      "
        f"{metrics['naive_brier_score']:.4f}"
    )

    print(
        f"Brier improvement:      "
        f"{metrics['brier_gain']:+.4f}"
    )

    if metrics["bce_gain"] > 0:
        print(
            "BCE conclusion: the model beats the naive "
            "prevalence predictor."
        )
    else:
        print(
            "BCE conclusion: the model does not beat the naive "
            "prevalence predictor."
        )


# ================================================================
# Label extraction
# ================================================================

def collect_dataset_labels(dataset):
    """
    Collects labels from a finite tf.data.Dataset.
    """

    labels = []

    for _, batch_labels in dataset:
        labels.append(
            np.asarray(batch_labels).ravel()
        )

    if not labels:
        raise ValueError(
            "No labels were found in the dataset."
        )

    return np.concatenate(labels).astype(np.int32)


train_labels = collect_dataset_labels(train_ds)

train_prevalence = float(
    np.mean(train_labels)
)

print(
    "\nTraining malignant prevalence used by the naive model: "
    f"{train_prevalence:.4f}"
)


# ================================================================
# Model
# ================================================================

def build_resnet50_transfer(
    input_shape=(PIXELS_H, PIXELS_W, 1),
    dropout_rate=0.5,
    spatial_dropout_rate=0.10,
):
    inputs = tf.keras.Input(
        shape=input_shape,
        name="mammogram_input",
    )

    # Applied only during training.
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                mode="horizontal",
                seed=SEED,
                name="random_horizontal_flip",
            ),
        ],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)

    # Grayscale image copied into three identical channels.
    x = tf.keras.layers.Concatenate(
        axis=-1,
        name="grayscale_to_rgb",
    )([x, x, x])

    # Keep this only when train_ds provides pixels in [0, 1].
    x = tf.keras.layers.Rescaling(
        scale=255.0,
        name="restore_255_scale",
    )(x)

    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(
            input_shape[0],
            input_shape[1],
            3,
        ),
    )

    base_model.trainable = False

    # Keeps BatchNormalization layers in inference mode.
    x = base_model(
        x,
        training=False,
    )

    # Spatial reduction:
    # approximately 12x12x2048 -> 3x3x2048 for a 384x384 input.
    x = tf.keras.layers.MaxPooling2D(
        pool_size=(4, 4),
        strides=(4, 4),
        padding="same",
        name="resnet_spatial_max_pooling",
    )(x)

    # Temporarily removes entire feature maps during training.
    x = tf.keras.layers.SpatialDropout2D(
        rate=spatial_dropout_rate,
        name="resnet_spatial_dropout",
    )(x)

    x = tf.keras.layers.Flatten(
        name="resnet_flatten",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name="classification_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-5
        ),
        name="classification_output",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="local_resnet50_transfer",
    )

    return model, base_model


model, base_model = build_resnet50_transfer(
    input_shape=(PIXELS_H, PIXELS_W, 1),
    dropout_rate=0.5,
    spatial_dropout_rate=0.10,
)


# ================================================================
# Compilation
# ================================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    ),

    # Optimisation objective:
    # BCE plus any regularisation penalties.
    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
        ),

        tf.keras.metrics.AUC(
            name="auc",
            curve="ROC",
        ),

        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),

        # BCE without L2 contribution.
        tf.keras.metrics.MeanMetricWrapper(
            raw_bce,
            name="raw_bce",
        ),

        tf.keras.metrics.Recall(
            name="recall_40",
            thresholds=0.40,
        ),
        tf.keras.metrics.Precision(
            name="precision_40",
            thresholds=0.40,
        ),

        tf.keras.metrics.Recall(
            name="recall_45",
            thresholds=0.45,
        ),
        tf.keras.metrics.Precision(
            name="precision_45",
            thresholds=0.45,
        ),

        tf.keras.metrics.Recall(
            name="recall_50",
            thresholds=0.50,
        ),
        tf.keras.metrics.Precision(
            name="precision_50",
            thresholds=0.50,
        ),

        tf.keras.metrics.Recall(
            name="recall_55",
            thresholds=0.55,
        ),
        tf.keras.metrics.Precision(
            name="precision_55",
            thresholds=0.55,
        ),
    ],
)


# ================================================================
# Run paths
# ================================================================

run_name = (
    f"resnet50_local_"
    f"{PIXELS_H}x{PIXELS_W}_"
    f"pool4_"
    f"spatialdrop010_"
    f"dropout050_"
    f"lr1e-4_"
    f"seed{SEED}"
)

head_checkpoint_path = (
    OUTPUT_MODEL / f"{run_name}.keras"
)

csv_log_path = (
    OUTPUT_MODEL / f"{run_name}.csv"
)


# ================================================================
# Callbacks
# ================================================================

callbacks_head = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        min_delta=1e-3,
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
        csv_log_path,
    ),
]


# ================================================================
# Training
# ================================================================

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_head,
)


# ================================================================
# Load best checkpoint
# ================================================================

best_model = tf.keras.models.load_model(
    head_checkpoint_path,
    compile=False,
)


# ================================================================
# Predictions
#
# During prediction:
# - dropout is disabled;
# - SpatialDropout2D is disabled;
# - RandomFlip is disabled.
# ================================================================

print("\nGenerating train predictions...")
train_prob, train_true = cnn_predict(
    best_model,
    train_ds,
)

print("Generating validation predictions...")
val_prob, val_true = cnn_predict(
    best_model,
    val_ds,
)

print("Generating test predictions...")
test_prob, test_true = cnn_predict(
    best_model,
    test_ds,
)


# ================================================================
# Probability reports
# ================================================================

train_report = calculate_probability_metrics(
    y_true=train_true,
    y_prob=train_prob,
    naive_probability=train_prevalence,
)

val_report = calculate_probability_metrics(
    y_true=val_true,
    y_prob=val_prob,
    naive_probability=train_prevalence,
)

test_report = calculate_probability_metrics(
    y_true=test_true,
    y_prob=test_prob,
    naive_probability=train_prevalence,
)

print_probability_report(
    "TRAIN — inference mode",
    train_report,
)

print_probability_report(
    "VALIDATION",
    val_report,
)

print_probability_report(
    "TEST",
    test_report,
)


# ================================================================
# Threshold metrics
#
# These are descriptive. The final threshold should ultimately be
# selected from validation probabilities rather than test results.
# ================================================================

print("\nValidation threshold metrics:")
evaluate_thresholds(
    val_prob,
    val_true,
)

print("\nTest threshold metrics:")
evaluate_thresholds(
    test_prob,
    test_true,
)


# ================================================================
# Summary table
# ================================================================

metrics_summary = pd.DataFrame(
    [
        {
            "dataset": "train",
            **train_report,
        },
        {
            "dataset": "validation",
            **val_report,
        },
        {
            "dataset": "test",
            **test_report,
        },
    ]
)

summary_path = (
    OUTPUT_MODEL / f"{run_name}_probability_metrics.csv"
)

metrics_summary.to_csv(
    summary_path,
    index=False,
)

print(
    f"\nProbability metrics saved to: {summary_path}"
)


# ================================================================
# Plot: total loss
#
# loss includes BCE + L2 regularisation.
# ================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_head.history["loss"],
    label="Train total loss",
)

plt.plot(
    history_head.history["val_loss"],
    label="Validation total loss",
)

plt.xlabel("Epoch")
plt.ylabel("BCE + regularisation")
plt.title("Training and validation total loss")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_PLOT / f"{run_name}_total_loss.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ================================================================
# Plot: raw BCE
#
# raw_bce excludes the L2 regularisation penalty.
# ================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_head.history["raw_bce"],
    label="Train raw BCE",
)

plt.plot(
    history_head.history["val_raw_bce"],
    label="Validation raw BCE",
)

plt.axhline(
    y=val_report["naive_bce"],
    linestyle="--",
    label=(
        "Naive validation BCE "
        f"({val_report['naive_bce']:.3f})"
    ),
)

plt.xlabel("Epoch")
plt.ylabel("Raw binary cross-entropy")
plt.title("Raw BCE versus naive predictor")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_PLOT / f"{run_name}_raw_bce.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ================================================================
# Plot: AUC
# ================================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_head.history["auc"],
    label="Train AUC",
)

plt.plot(
    history_head.history["val_auc"],
    label="Validation AUC",
)

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Training and validation AUC")
plt.legend()
plt.grid(True)

plt.savefig(
    OUTPUT_PLOT / f"{run_name}_auc.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()