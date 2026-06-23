import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    OUTPUT_NPY,
    OUTPUT_MODEL,
    PIXELS_H,
    PIXELS_W,
    SEED,
)
from src.functions import set_seed


# ============================================================
# CONFIGURATION
# ============================================================

ZOOM_TO_ROI = False
RESOLUTION = (PIXELS_H, PIXELS_W)

SAMPLES_PER_CLASS = 8
BATCH_SIZE_DEBUG = 4
EPOCHS_DEBUG = 50
LEARNING_RATE = 1e-3

# Arrête un test lorsque les 16 images sont parfaitement mémorisées.
TARGET_ACCURACY = 1.0
TARGET_LOSS = 1e-3

RESULTS_DIR = OUTPUT_MODEL / "ablation_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCTIBILITÉ
# ============================================================

def reset_environment(seed=SEED):
    """
    Nettoie la session Keras et réinitialise les générateurs aléatoires.
    """

    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(seed)


# ============================================================
# CHARGEMENT DU PETIT DATASET
# ============================================================

def load_balanced_debug_dataset(
    dataframe,
    samples_per_class=8,
):
    """
    Charge un échantillon fixe et équilibré directement depuis les fichiers
    .npy, sans passer par tf.data.
    """

    required_columns = {
        "set",
        "keep",
        "label",
        "preprocessed_image_path",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    train_df = dataframe.loc[
        (dataframe["set"] == "train")
        & (dataframe["keep"] == True)
    ].copy()

    train_df["label"] = train_df["label"].astype("int32")

    print("\nDistribution du train complet :")
    print(train_df["label"].value_counts().sort_index())

    available_labels = set(train_df["label"].unique())

    if available_labels != {0, 1}:
        raise ValueError(
            f"Les labels doivent être 0 et 1. Labels reçus : "
            f"{available_labels}"
        )

    selected_parts = []

    for label in [0, 1]:
        class_df = train_df.loc[
            train_df["label"] == label
        ]

        if len(class_df) < samples_per_class:
            raise ValueError(
                f"Classe {label} : seulement {len(class_df)} images "
                f"disponibles pour {samples_per_class} demandées."
            )

        selected_parts.append(
            class_df.sample(
                n=samples_per_class,
                random_state=SEED,
            )
        )

    debug_df = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # Mélange fixe pour que chaque architecture reçoive exactement les
    # mêmes images dans le même ordre.
    debug_df = debug_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    images = []
    labels = []
    paths = []

    expected_shape = (
        PIXELS_H,
        PIXELS_W,
        1,
    )

    for index, row in debug_df.iterrows():
        image_path = Path(
            str(row["preprocessed_image_path"])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {image_path}"
            )

        image = np.load(
            image_path
        ).astype("float32")

        if image.shape == (PIXELS_H, PIXELS_W):
            image = image[..., np.newaxis]

        if image.shape != expected_shape:
            raise ValueError(
                f"Shape incorrecte pour {image_path}: "
                f"{image.shape}, attendu={expected_shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError(
                f"NaN ou Inf dans {image_path}"
            )

        label = int(row["label"])

        images.append(image)
        labels.append(label)
        paths.append(str(image_path))

        print(
            f"{index + 1:02d} | "
            f"label={label} | "
            f"min={image.min():.6f} | "
            f"max={image.max():.6f} | "
            f"mean={image.mean():.6f} | "
            f"std={image.std():.6f}"
        )

    x_debug = np.stack(images).astype("float32")
    y_debug = np.asarray(labels).astype("float32")

    return x_debug, y_debug, paths


# ============================================================
# CONTRÔLE DES DONNÉES
# ============================================================

def validate_debug_dataset(
    images,
    labels,
    paths,
):
    print("\n" + "=" * 72)
    print("VALIDATION DU DATASET")
    print("=" * 72)

    print("Images :", images.shape, images.dtype)
    print("Labels :", labels.shape, labels.dtype)

    unique_labels, label_counts = np.unique(
        labels,
        return_counts=True,
    )

    print(
        "Distribution :",
        dict(zip(unique_labels, label_counts)),
    )

    print(
        f"Valeurs : min={images.min():.6f}, "
        f"max={images.max():.6f}, "
        f"mean={images.mean():.6f}, "
        f"std={images.std():.6f}"
    )

    if len(images) != len(labels):
        raise ValueError(
            "Le nombre d'images ne correspond pas au nombre de labels."
        )

    if set(unique_labels) != {0.0, 1.0}:
        raise ValueError(
            f"Labels invalides : {unique_labels}"
        )

    hashes = [
        hashlib.md5(image.tobytes()).hexdigest()
        for image in images
    ]

    print(
        f"Images uniques : {len(set(hashes))}/{len(hashes)}"
    )

    hash_to_labels = {}

    for image_hash, label in zip(hashes, labels):
        hash_to_labels.setdefault(
            image_hash,
            set(),
        ).add(int(label))

    conflicts = {
        image_hash: associated_labels
        for image_hash, associated_labels
        in hash_to_labels.items()
        if len(associated_labels) > 1
    }

    if conflicts:
        raise ValueError(
            f"Images identiques avec labels contradictoires : {conflicts}"
        )

    print("Premiers fichiers :")

    for path, label in zip(paths[:5], labels[:5]):
        print(f"  label={int(label)} | {path}")


# ============================================================
# BLOC CONVOLUTIONNEL
# ============================================================

def convolutional_backbone(
    inputs,
    use_batch_norm,
    use_l2,
):
    """
    Reproduit le backbone de ton architecture normale.
    """

    x = inputs

    regularizer = (
        tf.keras.regularizers.l2(1e-4)
        if use_l2
        else None
    )

    for filters in [16, 32, 64]:
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=not use_batch_norm,
            kernel_regularizer=regularizer,
        )(x)

        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=2
        )(x)

    return x


# ============================================================
# CONSTRUCTION PARAMÉTRABLE DU MODÈLE
# ============================================================

def build_test_model(
    input_shape,
    pooling_mode="average",
    use_batch_norm=True,
    use_l2=True,
    dropout_rate=0.5,
):
    """
    pooling_mode:
        - "average"
        - "maximum"
        - "average_maximum"
        - "flatten"
    """

    inputs = tf.keras.Input(
        shape=input_shape
    )

    x = convolutional_backbone(
        inputs=inputs,
        use_batch_norm=use_batch_norm,
        use_l2=use_l2,
    )

    if pooling_mode == "average":
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    elif pooling_mode == "maximum":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)

    elif pooling_mode == "average_maximum":
        average = (
            tf.keras.layers.GlobalAveragePooling2D()(x)
        )

        maximum = (
            tf.keras.layers.GlobalMaxPooling2D()(x)
        )

        x = tf.keras.layers.Concatenate()(
            [average, maximum]
        )

    elif pooling_mode == "flatten":
        x = tf.keras.layers.Flatten()(x)

    else:
        raise ValueError(
            f"Pooling inconnu : {pooling_mode}"
        )

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(
            dropout_rate
        )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )


# ============================================================
# CALLBACK D'ARRÊT POUR MÉMORISATION
# ============================================================

class StopWhenMemorized(tf.keras.callbacks.Callback):
    """
    Arrête l'entraînement lorsque les images sont mémorisées.
    """

    def __init__(
        self,
        target_accuracy=1.0,
        target_loss=1e-3,
    ):
        super().__init__()

        self.target_accuracy = target_accuracy
        self.target_loss = target_loss

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        accuracy = logs.get("accuracy")
        loss = logs.get("loss")

        if accuracy is None or loss is None:
            return

        if (
            accuracy >= self.target_accuracy
            and loss <= self.target_loss
        ):
            print(
                "\nDataset mémorisé : "
                f"accuracy={accuracy:.4f}, "
                f"loss={loss:.6f}"
            )

            self.model.stop_training = True


# ============================================================
# EXÉCUTION D'UN TEST
# ============================================================

def run_ablation_test(
    test_name,
    config,
    images,
    labels,
):
    print("\n\n" + "#" * 80)
    print(f"TEST : {test_name}")
    print("#" * 80)

    print(
        json.dumps(
            config,
            indent=2,
        )
    )

    reset_environment()

    model = build_test_model(
        input_shape=(
            PIXELS_H,
            PIXELS_W,
            1,
        ),
        **config,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    print(
        f"Paramètres : {model.count_params():,}"
    )

    probabilities_before = model.predict(
        images,
        verbose=0,
    ).reshape(-1)

    initial_weights = [
        weight.copy()
        for weight in model.get_weights()
    ]

    history = model.fit(
        x=images,
        y=labels,
        batch_size=BATCH_SIZE_DEBUG,
        epochs=EPOCHS_DEBUG,
        shuffle=False,
        verbose=0,
        callbacks=[
            StopWhenMemorized(
                target_accuracy=TARGET_ACCURACY,
                target_loss=TARGET_LOSS,
            )
        ],
    )

    final_weights = model.get_weights()

    probabilities_after = model.predict(
        images,
        verbose=0,
    ).reshape(-1)

    evaluation = model.evaluate(
        images,
        labels,
        batch_size=BATCH_SIZE_DEBUG,
        verbose=0,
        return_dict=True,
    )

    weight_changes = [
        float(
            np.mean(
                np.abs(
                    final_weight
                    - initial_weight
                )
            )
        )
        for initial_weight, final_weight
        in zip(
            initial_weights,
            final_weights,
        )
    ]

    predicted_labels = (
        probabilities_after >= 0.5
    ).astype("int32")

    epochs_completed = len(
        history.history["loss"]
    )

    result = {
        "test_name": test_name,
        "pooling_mode": config["pooling_mode"],
        "use_batch_norm": config["use_batch_norm"],
        "use_l2": config["use_l2"],
        "dropout_rate": config["dropout_rate"],
        "parameters": model.count_params(),
        "epochs_completed": epochs_completed,
        "final_loss": float(evaluation["loss"]),
        "final_accuracy": float(evaluation["accuracy"]),
        "final_auc": float(evaluation["auc"]),
        "minimum_probability": float(
            probabilities_after.min()
        ),
        "maximum_probability": float(
            probabilities_after.max()
        ),
        "probability_std": float(
            probabilities_after.std()
        ),
        "maximum_weight_change": max(
            weight_changes
        ),
        "memorized": bool(
            evaluation["accuracy"] >= 0.999
        ),
    }

    print("\nRésultat :")

    print(
        f"  epochs     : {epochs_completed}"
    )
    print(
        f"  loss       : {result['final_loss']:.6f}"
    )
    print(
        f"  accuracy   : {result['final_accuracy']:.4f}"
    )
    print(
        f"  AUC        : {result['final_auc']:.4f}"
    )
    print(
        f"  prob min   : "
        f"{result['minimum_probability']:.6f}"
    )
    print(
        f"  prob max   : "
        f"{result['maximum_probability']:.6f}"
    )
    print(
        f"  prob std   : "
        f"{result['probability_std']:.6f}"
    )
    print(
        f"  mémorisé   : {result['memorized']}"
    )

    print("\nPrédictions finales :")

    for index, (
        true_label,
        probability,
        predicted_label,
    ) in enumerate(
        zip(
            labels,
            probabilities_after,
            predicted_labels,
        )
    ):
        print(
            f"  {index:02d} | "
            f"true={int(true_label)} | "
            f"prob={probability:.6f} | "
            f"pred={predicted_label}"
        )

    history_df = pd.DataFrame(
        history.history
    )

    safe_test_name = (
        test_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
    )

    history_df.to_csv(
        RESULTS_DIR
        / f"history_{safe_test_name}.csv",
        index=False,
    )

    model.save(
        RESULTS_DIR
        / f"model_{safe_test_name}.keras"
    )

    del model
    gc.collect()

    return result


# ============================================================
# SCRIPT PRINCIPAL
# ============================================================

if __name__ == "__main__":

    reset_environment()

    if ZOOM_TO_ROI:
        dataset_folder = (
            f"zoom_{RESOLUTION[0]}x{RESOLUTION[1]}"
        )
    else:
        dataset_folder = (
            f"full_{RESOLUTION[0]}x{RESOLUTION[1]}"
        )

    csv_path = (
        OUTPUT_NPY
        / f"dataset_index_{dataset_folder}.csv"
    )

    print("CSV :", csv_path)

    df = pd.read_csv(csv_path)

    print("Dimensions CSV :", df.shape)

    x_debug, y_debug, debug_paths = (
        load_balanced_debug_dataset(
            dataframe=df,
            samples_per_class=SAMPLES_PER_CLASS,
        )
    )

    validate_debug_dataset(
        images=x_debug,
        labels=y_debug,
        paths=debug_paths,
    )

    # Chaque test ne modifie qu'un ou quelques éléments.
    tests = {
        "01_baseline_originale": {
            "pooling_mode": "average",
            "use_batch_norm": True,
            "use_l2": True,
            "dropout_rate": 0.5,
        },

        "02_sans_l2": {
            "pooling_mode": "average",
            "use_batch_norm": True,
            "use_l2": False,
            "dropout_rate": 0.5,
        },

        "03_sans_dropout": {
            "pooling_mode": "average",
            "use_batch_norm": True,
            "use_l2": True,
            "dropout_rate": 0.0,
        },

        "04_sans_l2_sans_dropout": {
            "pooling_mode": "average",
            "use_batch_norm": True,
            "use_l2": False,
            "dropout_rate": 0.0,
        },

        "05_sans_batch_norm": {
            "pooling_mode": "average",
            "use_batch_norm": False,
            "use_l2": False,
            "dropout_rate": 0.0,
        },

        "06_global_max_pooling": {
            "pooling_mode": "maximum",
            "use_batch_norm": False,
            "use_l2": False,
            "dropout_rate": 0.0,
        },

        "07_average_plus_maximum": {
            "pooling_mode": "average_maximum",
            "use_batch_norm": False,
            "use_l2": False,
            "dropout_rate": 0.0,
        },

        "08_flatten": {
            "pooling_mode": "flatten",
            "use_batch_norm": False,
            "use_l2": False,
            "dropout_rate": 0.0,
        },
    }

    all_results = []

    for test_name, test_config in tests.items():
        try:
            result = run_ablation_test(
                test_name=test_name,
                config=test_config,
                images=x_debug,
                labels=y_debug,
            )

            all_results.append(result)

        except Exception as error:
            print(
                f"\nERREUR pendant {test_name}: "
                f"{type(error).__name__}: {error}"
            )

            all_results.append({
                "test_name": test_name,
                "error": str(error),
                "memorized": False,
            })

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df.sort_values(
        by=[
            "final_accuracy",
            "final_loss",
        ],
        ascending=[
            False,
            True,
        ],
        na_position="last",
    )

    results_path = (
        RESULTS_DIR
        / "ablation_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n\n" + "=" * 100)
    print("RÉSUMÉ DES TESTS")
    print("=" * 100)

    columns_to_display = [
        column
        for column in [
            "test_name",
            "pooling_mode",
            "use_batch_norm",
            "use_l2",
            "dropout_rate",
            "parameters",
            "epochs_completed",
            "final_loss",
            "final_accuracy",
            "final_auc",
            "probability_std",
            "memorized",
        ]
        if column in results_df.columns
    ]

    print(
        results_df[
            columns_to_display
        ].to_string(
            index=False
        )
    )

    print(
        "\nRésumé enregistré dans :",
        results_path,
    )