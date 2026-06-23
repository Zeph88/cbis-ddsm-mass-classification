
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    OUTPUT_NPY,
    SEED,
)
from src.functions import (
    evaluate_thresholds,
    set_seed,
    train_val_test_sets,
)


# ============================================================
# Configuration
# ============================================================

RESOLUTION = (598, 598)
INPUT_SHAPE = (*RESOLUTION, 1)

ZOOM_TO_ROI = True

EPOCHS = 50
LEARNING_RATE = 1e-4

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 3

RUN_GAP_MODEL = True
RUN_SPATIAL_MODEL = True

MODEL_OUTPUT_DIR = Path("trained_models")
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load dataset index
# ============================================================

zoom_path = (
    f"zoom_{RESOLUTION[0]}x{RESOLUTION[1]}"
    if ZOOM_TO_ROI
    else f"full_{RESOLUTION[0]}x{RESOLUTION[1]}"
)

dataset_index_path = (
    OUTPUT_NPY / f"dataset_index_{zoom_path}.csv"
)

print(f"Loading dataset index: {dataset_index_path}")

if not dataset_index_path.exists():
    raise FileNotFoundError(
        f"Dataset index not found: {dataset_index_path}"
    )

df = pd.read_csv(dataset_index_path)

required_columns = {
    "set",
    "label",
    "preprocessed_image_path",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


print("\nDataset distribution")
print("--------------------")
print(
    df.groupby("set")["label"]
    .agg(
        sample_count="count",
        positive_ratio="mean",
    )
)


# ============================================================
# Build TensorFlow datasets
# ============================================================

train_ds, val_ds, test_ds = train_val_test_sets(
    df,
    BATCH_SIZE,
    SEED,
)


# ============================================================
# Dataset validation
# ============================================================

def validate_dataset_shape(
    dataset: tf.data.Dataset,
    dataset_name: str,
    expected_shape: tuple[int, int, int],
) -> None:
    """
    Verify the actual image shape returned by the TensorFlow pipeline.
    """

    for images, labels in dataset.take(1):
        actual_shape = tuple(images.shape[1:])

        print(f"\n{dataset_name}")
        print("-" * len(dataset_name))
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(
            f"Pixel range: "
            f"{tf.reduce_min(images).numpy():.6f} "
            f"to {tf.reduce_max(images).numpy():.6f}"
        )

        if actual_shape != expected_shape:
            raise ValueError(
                f"{dataset_name} returned image shape "
                f"{actual_shape}, expected {expected_shape}."
            )

        return

    raise ValueError(f"{dataset_name} is empty.")


validate_dataset_shape(
    train_ds,
    "Train dataset",
    INPUT_SHAPE,
)

validate_dataset_shape(
    val_ds,
    "Validation dataset",
    INPUT_SHAPE,
)

validate_dataset_shape(
    test_ds,
    "Test dataset",
    INPUT_SHAPE,
)


# ============================================================
# Shared convolutional backbone
# ============================================================

def build_backbone(
    inputs: tf.Tensor,
) -> tf.Tensor:
    """
    Shared feature extractor used by both experimental models.

    Both models therefore differ only in their spatial aggregation head.
    """

    x = inputs

    filter_sequence = [
        16,
        32,
        64,
        128,
        128,
        128,
    ]

    for block_index, filters in enumerate(
        filter_sequence,
        start=1,
    ):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
            kernel_initializer="he_normal",
            name=f"conv_{block_index}",
        )(x)

        x = tf.keras.layers.ReLU(
            name=f"relu_{block_index}",
        )(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            name=f"pool_{block_index}",
        )(x)

    # Approximate spatial evolution:
    # 598 -> 299 -> 149 -> 74 -> 37 -> 18 -> 9
    #
    # Output shape:
    # approximately 9 x 9 x 128

    return x


# ============================================================
# Model builder
# ============================================================

def build_crop_model(
    head_type: str,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    dropout_rate: float = 0.5,
) -> tf.keras.Model:
    """
    Build either:

    head_type="gap":
        1x1 convolution -> GlobalAveragePooling2D -> Dense

    head_type="spatial":
        1x1 convolution -> Flatten -> Dense

    The 1x1 convolution is shared conceptually by both models so that
    the main experimental difference is GAP versus Flatten.
    """

    if head_type not in {"gap", "spatial"}:
        raise ValueError(
            "head_type must be either 'gap' or 'spatial'."
        )

    inputs = tf.keras.Input(
        shape=input_shape,
        name="local_crop",
    )

    x = build_backbone(inputs)

    # Reduce the number of channels before aggregation.
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        name="channel_reduction",
    )(x)

    if head_type == "gap":
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="global_average_pooling",
        )(x)

    else:
        x = tf.keras.layers.Flatten(
            name="spatial_flatten",
        )(x)

    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-4
        ),
        name="dense_features",
    )(x)

    x = tf.keras.layers.Dropout(
        rate=dropout_rate,
        name="dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="prediction",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"crop598_{head_type}_model",
    )


# ============================================================
# Compilation
# ============================================================

def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy",
            ),
            tf.keras.metrics.AUC(
                name="auc",
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
                name="recall_50",
                thresholds=0.50,
            ),
            tf.keras.metrics.Precision(
                name="precision_50",
                thresholds=0.50,
            ),
        ],
    )


# ============================================================
# Prediction diagnostics
# ============================================================

def inspect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    labels = []

    for images, batch_labels in dataset:
        batch_probabilities = model(
            images,
            training=False,
        ).numpy().reshape(-1)

        probabilities.append(
            batch_probabilities
        )

        labels.append(
            batch_labels.numpy().reshape(-1)
        )

    if not probabilities:
        raise ValueError(
            f"{dataset_name} is empty."
        )

    probabilities = np.concatenate(
        probabilities
    )

    labels = np.concatenate(
        labels
    )

    quantiles = np.quantile(
        probabilities,
        [
            0.00,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.00,
        ],
    )

    print(f"\n{dataset_name} predictions")
    print("-" * (len(dataset_name) + 12))
    print(f"Samples: {len(probabilities)}")
    print(f"Minimum: {probabilities.min():.6f}")
    print(f"Maximum: {probabilities.max():.6f}")
    print(f"Mean: {probabilities.mean():.6f}")
    print(
        f"Standard deviation: "
        f"{probabilities.std():.6f}"
    )
    print(f"Quantiles: {quantiles}")

    return probabilities, labels


# ============================================================
# Callbacks
# ============================================================

def create_callbacks(
    checkpoint_path: Path,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=0.002,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_delta=0.002,
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            filename=checkpoint_path.with_suffix(
                ".csv"
            ),
            append=False,
        ),
    ]


# ============================================================
# Run one experiment
# ============================================================

def run_experiment(
    head_type: str,
) -> dict:
    """
    Train, save and evaluate one model.

    A fresh seed and fresh model are used for every run.
    """

    print("\n")
    print("=" * 70)
    print(f"Running experiment: {head_type.upper()}")
    print("=" * 70)

    # Reset graph state and random generators.
    tf.keras.backend.clear_session()
    set_seed(SEED)

    model = build_crop_model(
        head_type=head_type,
        input_shape=INPUT_SHAPE,
        dropout_rate=0.5,
    )

    compile_model(
        model=model,
        learning_rate=LEARNING_RATE,
    )

    model.summary()

    checkpoint_path = (
        MODEL_OUTPUT_DIR
        / f"model_local_crop598_{head_type}.keras"
    )

    callbacks = create_callbacks(
        checkpoint_path=checkpoint_path,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not created: "
            f"{checkpoint_path}"
        )

    best_model = tf.keras.models.load_model(
        checkpoint_path
    )

    print(
        f"\nEvaluating best {head_type} model"
    )

    train_probabilities, train_labels = (
        inspect_predictions(
            model=best_model,
            dataset=train_ds,
            dataset_name=f"{head_type.upper()} train",
        )
    )

    val_probabilities, val_labels = (
        inspect_predictions(
            model=best_model,
            dataset=val_ds,
            dataset_name=f"{head_type.upper()} validation",
        )
    )

    test_probabilities, test_labels = (
        inspect_predictions(
            model=best_model,
            dataset=test_ds,
            dataset_name=f"{head_type.upper()} test",
        )
    )

    print(
        f"\nThreshold evaluation: "
        f"{head_type.upper()}"
    )

    evaluate_thresholds(
        test_probabilities,
        test_labels,
    )

    train_evaluation = best_model.evaluate(
        train_ds,
        verbose=0,
        return_dict=True,
    )

    validation_evaluation = best_model.evaluate(
        val_ds,
        verbose=0,
        return_dict=True,
    )

    test_evaluation = best_model.evaluate(
        test_ds,
        verbose=0,
        return_dict=True,
    )

    result = {
        "head_type": head_type,
        "best_epoch": (
            int(np.argmax(history.history["val_auc"]))
            + 1
        ),
        "train_auc": train_evaluation["auc"],
        "validation_auc": validation_evaluation["auc"],
        "test_auc": test_evaluation["auc"],
        "train_loss": train_evaluation["loss"],
        "validation_loss": validation_evaluation["loss"],
        "test_loss": test_evaluation["loss"],
        "test_accuracy": test_evaluation["accuracy"],
        "test_probability_min": float(
            test_probabilities.min()
        ),
        "test_probability_max": float(
            test_probabilities.max()
        ),
        "test_probability_mean": float(
            test_probabilities.mean()
        ),
        "test_probability_std": float(
            test_probabilities.std()
        ),
        "parameter_count": best_model.count_params(),
        "checkpoint": str(checkpoint_path),
    }

    return result


# ============================================================
# Run requested experiments
# ============================================================

experiment_results = []

if RUN_GAP_MODEL:
    gap_result = run_experiment(
        head_type="gap",
    )

    experiment_results.append(
        gap_result
    )


if RUN_SPATIAL_MODEL:
    spatial_result = run_experiment(
        head_type="spatial",
    )

    experiment_results.append(
        spatial_result
    )


# ============================================================
# Compare results
# ============================================================

if experiment_results:
    results_df = pd.DataFrame(
        experiment_results
    )

    results_output_path = (
        MODEL_OUTPUT_DIR
        / "crop598_head_comparison.csv"
    )

    results_df.to_csv(
        results_output_path,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    columns_to_display = [
        "head_type",
        "best_epoch",
        "parameter_count",
        "train_auc",
        "validation_auc",
        "test_auc",
        "test_accuracy",
        "test_probability_min",
        "test_probability_max",
        "test_probability_std",
    ]

    print(
        results_df[
            columns_to_display
        ].to_string(index=False)
    )

    print(
        f"\nSaved comparison to: "
        f"{results_output_path}"
    )

