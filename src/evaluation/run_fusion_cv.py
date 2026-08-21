import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.config import BATCH_SIZE, EPOCHS, GLOBAL_HEIGHT, GLOBAL_WIDTH, LOCAL_HEIGHT, LOCAL_WIDTH, OUTPUT_MODEL, OUTPUT_NPY, SEED, SPLITS_DIR, MAMMOGRAM_KEY, N_OUTER_FOLDS
from src.modeling.local_resnet50 import build_local_model
from src.modeling.global_resnet50 import build_global_model
from src.modeling.fusion import build_residual_fusion, build_symmetric_fusion
from src.functions import ensure_directory, set_seed, load_data, parse_arguments
from src.training.dataset_preparation import build_tf_dataset
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions

N_INNER_FOLDS = N_OUTER_FOLDS

CV_ROOT = OUTPUT_MODEL / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{SEED}"
FOLDS_DIR = CV_ROOT / "folds"
RESULTS_DIR = CV_ROOT / "results"

ensure_directory(CV_ROOT)
ensure_directory(FOLDS_DIR)
ensure_directory(RESULTS_DIR)

def compile_binary_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR")
        ]
    )

    return model

def create_outer_patient_folds(dev_meta):
    splitter = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
    labels = (dev_meta["label"].to_numpy())
    groups = (dev_meta["patient_id"].to_numpy())
    dummy_x = np.zeros((len(dev_meta), 1), dtype=np.float32)

    patient_to_fold = {}
    rows = []

    for fold, (_, eval_indices) in enumerate(splitter.split(dummy_x, labels, groups)):
        patients = pd.unique(groups[eval_indices])

        for patient_id in patients:
            if patient_id in patient_to_fold:
                raise RuntimeError(f"{patient_id} assigned twice.")

            patient_to_fold[patient_id] = fold

            rows.append(
                {
                    "patient_id": patient_id,
                    "outer_fold": fold,
                }
            )

    pd.DataFrame(rows).to_csv(
        RESULTS_DIR
        / "outer_patient_folds.csv",
        index=False,
    )

    return patient_to_fold

def create_inner_patient_split(
    dev_meta,
    patient_to_outer_fold,
    outer_fold,
):
    outer_train_meta = dev_meta[
        dev_meta["patient_id"].map(
            patient_to_outer_fold
        )
        != outer_fold
    ].copy()

    splitter = StratifiedGroupKFold(
        n_splits=N_INNER_FOLDS,
        shuffle=True,
        random_state=(
            SEED
            + outer_fold
        ),
    )

    labels = (
        outer_train_meta["label"]
        .to_numpy()
    )

    groups = (
        outer_train_meta["patient_id"]
        .to_numpy()
    )

    dummy_x = np.zeros(
        (len(outer_train_meta), 1),
        dtype=np.float32,
    )

    train_indices, val_indices = next(
        splitter.split(
            dummy_x,
            labels,
            groups,
        )
    )

    inner_train_patients = set(
        groups[train_indices]
    )

    inner_val_patients = set(
        groups[val_indices]
    )

    outer_eval_patients = {
        patient_id
        for patient_id, fold
        in patient_to_outer_fold.items()
        if fold == outer_fold
    }

    if (
        inner_train_patients
        & inner_val_patients
    ):
        raise RuntimeError(
            "Inner train/validation leakage."
        )

    if (
        outer_eval_patients
        & inner_train_patients
    ):
        raise RuntimeError(
            "Outer fold leaked into inner train."
        )

    if (
        outer_eval_patients
        & inner_val_patients
    ):
        raise RuntimeError(
            "Outer fold leaked into inner validation."
        )

    return {
        "inner_train": inner_train_patients,
        "inner_validation": inner_val_patients,
        "outer_evaluation": outer_eval_patients,
    }

def load_preprocessed_indexes(dev_patient_ids):

    local_path = OUTPUT_NPY / f"dataset_index_zoom_{LOCAL_HEIGHT}x{LOCAL_WIDTH}.csv"
    global_path = OUTPUT_NPY / f"dataset_index_full_{GLOBAL_HEIGHT}x{GLOBAL_WIDTH}.csv"
    local_df, global_df = load_data(local_path, global_path)

    local_df["patient_id"] = (
        local_df["patient_id"]
        .astype(str)
    )

    global_df["patient_id"] = (
        global_df["patient_id"]
        .astype(str)
    )

    local_df = local_df[
        local_df["patient_id"].isin(
            dev_patient_ids
        )
    ].copy()

    global_df = global_df[
        global_df["patient_id"].isin(
            dev_patient_ids
        )
    ].copy()

    return (
        local_df,
        global_df,
    )

def assign_partition(
    df,
    patient_partition,
):
    mapping = {}

    for patient_id in patient_partition[
        "inner_train"
    ]:
        mapping[patient_id] = "train"

    for patient_id in patient_partition[
        "inner_validation"
    ]:
        mapping[patient_id] = "validation"

    for patient_id in patient_partition[
        "outer_evaluation"
    ]:
        mapping[
            patient_id
        ] = "outer_evaluation"

    output = df.copy()

    output["cv_set"] = (
        output["patient_id"]
        .map(mapping)
    )

    output = output[
        output["cv_set"].notna()
    ].copy()

    return output

def build_single_input_datasets(
    df,
    height,
    width,
):
    train_df = df[
        df["cv_set"] == "train"
    ]

    val_df = df[
        df["cv_set"] == "validation"
    ]

    train_ds = build_tf_dataset(
        train_df,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        image_height=height,
        image_width=width,
    )

    val_ds = build_tf_dataset(
        val_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        image_height=height,
        image_width=width,
    )

    return (
        train_ds,
        val_ds,
    )

def build_paired_dataframe(
    local_df,
    global_df,
):
    local_df = local_df.copy()
    global_df = global_df.copy()

    local_df[
        "local_path"
    ] = local_df[
        "preprocessed_image_path"
    ]

    global_lookup = (
        global_df[
            MAMMOGRAM_KEY
            + [
                "preprocessed_image_path",
                "cv_set",
            ]
        ]
        .drop_duplicates(
            subset=MAMMOGRAM_KEY
        )
        .rename(
            columns={
                "preprocessed_image_path":
                    "global_path",
                "cv_set":
                    "global_cv_set",
            }
        )
    )

    paired = local_df.merge(
        global_lookup,
        on=MAMMOGRAM_KEY,
        how="inner",
        validate="many_to_one",
    )

    mismatch = (
        paired["cv_set"]
        != paired["global_cv_set"]
    )

    if mismatch.any():
        raise RuntimeError(
            "Local/global partition mismatch."
        )

    return paired

def build_paired_datasets(
    paired_df,
):
    train_df = paired_df[
        paired_df["cv_set"]
        == "train"
    ]

    val_df = paired_df[
        paired_df["cv_set"]
        == "validation"
    ]

    outer_df = paired_df[
        paired_df["cv_set"]
        == "outer_evaluation"
    ]

    kwargs = dict(
        batch_size=BATCH_SIZE,
        seed=SEED,
        path_image="local_path",
        added_path_image="global_path",
        image_height=LOCAL_HEIGHT,
        image_width=LOCAL_WIDTH,
        added_image_height=GLOBAL_HEIGHT,
        added_image_width=GLOBAL_WIDTH,
    )

    train_ds = build_tf_dataset(
        train_df,
        shuffle=True,
        **kwargs,
    )

    val_ds = build_tf_dataset(
        val_df,
        shuffle=False,
        **kwargs,
    )

    outer_ds = build_tf_dataset(
        outer_df,
        shuffle=False,
        **kwargs,
    )

    return (
        train_ds,
        val_ds,
        outer_ds,
        outer_df.reset_index(
            drop=True
        ),
    )

def callbacks_for(
    checkpoint_path,
    log_path,
):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=False,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
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
            filename=str(
                log_path
            ),
        ),
    ]

def train_branch(
    build_fn,
    train_ds,
    val_ds,
    checkpoint_path,
    log_path,
    force=False,
):
    if (
        checkpoint_path.exists()
        and not force
    ):
        print(
            "Reuse:",
            checkpoint_path,
        )
        return

    set_seed(
        SEED
    )

    model = build_fn(
        seed=SEED
    )

    compile_binary_model(
        model,
        learning_rate=1e-4,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_for(
            checkpoint_path,
            log_path,
        ),
        verbose=2,
    )

    del model

    tf.keras.backend.clear_session()
    gc.collect()

def train_fusion(
    architecture,
    local_model_path,
    global_model_path,
    train_ds,
    val_ds,
    checkpoint_path,
    log_path,
    force=False,
):
    if (
        checkpoint_path.exists()
        and not force
    ):
        print(
            "Reuse:",
            checkpoint_path,
        )
        return

    tf.keras.backend.clear_session()
    gc.collect()

    set_seed(
        SEED
    )

    local_model = (
        tf.keras.models.load_model(
            local_model_path,
            compile=False,
        )
    )

    global_model = (
        tf.keras.models.load_model(
            global_model_path,
            compile=False,
        )
    )

    if architecture == "symmetric":
        model = build_symmetric_fusion(
            local_model,
            global_model,
        )

    elif architecture == "residual":
        model = build_residual_fusion(
            local_model,
            global_model,
        )

    else:
        raise ValueError(
            architecture
        )

    compile_binary_model(
        model,
        learning_rate=1e-4,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_for(
            checkpoint_path,
            log_path,
        ),
        verbose=2,
    )

    del model
    del local_model
    del global_model

    tf.keras.backend.clear_session()
    gc.collect()

def evaluate_fusion(model_path, outer_eval_ds, outer_metadata):

    model = tf.keras.models.load_model(model_path, compile=False)

    y_true, probabilities = collect_binary_predictions(model, outer_eval_ds)

    scores = calculate_metrics(
        y_true,
        probabilities,
    )

    del model

    tf.keras.backend.clear_session()
    gc.collect()

    return (
        probabilities,
        scores,
    )

def run_one_fold(
    outer_fold,
    dev_meta,
    patient_to_outer_fold,
    local_dev,
    global_dev,
    force=False,
):
    fold_dir = (
        FOLDS_DIR
        / f"fold_{outer_fold}"
    )

    ensure_directory(
        fold_dir
    )

    patient_partition = (
        create_inner_patient_split(
            dev_meta,
            patient_to_outer_fold,
            outer_fold,
        )
    )

    local_fold = assign_partition(
        local_dev,
        patient_partition,
    )

    global_fold = assign_partition(
        global_dev,
        patient_partition,
    )

    # ----------------------------
    # LOCAL
    # ----------------------------

    (
        local_train_ds,
        local_val_ds,
    ) = build_single_input_datasets(
        local_fold,
        LOCAL_HEIGHT,
        LOCAL_WIDTH,
    )

    local_path = fold_dir / "local.keras"

    train_branch(
        build_local_model,
        local_train_ds,
        local_val_ds,
        local_path,
        fold_dir
        / "local_training.csv",
        force=force,
    )

    del (
        local_train_ds,
        local_val_ds,
    )

    # ----------------------------
    # GLOBAL
    # ----------------------------

    (
        global_train_ds,
        global_val_ds,
    ) = build_single_input_datasets(
        global_fold,
        GLOBAL_HEIGHT,
        GLOBAL_WIDTH,
    )

    global_path = (
        fold_dir
        / "global.keras"
    )

    train_branch(
        build_global_model,
        global_train_ds,
        global_val_ds,
        global_path,
        fold_dir
        / "global_training.csv",
        force=force,
    )

    del (
        global_train_ds,
        global_val_ds,
    )

    gc.collect()

    # ----------------------------
    # FUSION DATASET
    # ----------------------------

    paired = build_paired_dataframe(
        local_fold,
        global_fold,
    )

    (
        fusion_train_ds,
        fusion_val_ds,
        outer_ds,
        outer_metadata,
    ) = build_paired_datasets(
        paired
    )

    predictions_df = outer_metadata[
        [
            "sample_id",
            "patient_id",
            "left or right breast",
            "image view",
            "label",
        ]
    ].copy()

    predictions_df[
        "fold"
    ] = outer_fold

    predictions_df[
        "seed"
    ] = SEED

    metric_rows = []

    # ----------------------------
    # SYMMETRIC + RESIDUAL
    # ----------------------------

    for architecture in [
        "symmetric",
        "residual",
    ]:
        fusion_path = (
            fold_dir
            / f"{architecture}.keras"
        )

        train_fusion(
            architecture,
            local_path,
            global_path,
            fusion_train_ds,
            fusion_val_ds,
            fusion_path,
            fold_dir
            / f"{architecture}_training.csv",
            force=force,
        )

        (
            probabilities,
            scores,
        ) = evaluate_fusion(
            fusion_path,
            outer_ds,
            outer_metadata,
        )

        predictions_df[
            f"{architecture}_probability"
        ] = probabilities

        metric_rows.append(
            {
                "fold": outer_fold,
                "seed": SEED,
                "architecture":
                    architecture,
                "n_outer_evaluation":
                    len(outer_metadata),
                "n_outer_evaluation_patients":
                    outer_metadata[
                        "patient_id"
                    ].nunique(),
                **scores,
            }
        )

        print(
            architecture,
            scores,
        )

    predictions_df.to_csv(
        fold_dir
        / "oof_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        fold_dir
        / "fold_metrics.csv",
        index=False,
    )

def main():
    args = parse_arguments(
        description="Generate fold predictions.",
        arguments=[
            {
                "name": "--force",
                "action": "store_true",
            },
        ],
        exclusive_arguments=[
            {
                "name": "--fold",
                "type": int,
                "choices": range(N_OUTER_FOLDS),
            },
            {
                "name": "--all",
                "action": "store_true",
            },
        ],
        exclusive_required=True,
    )
    set_seed(SEED)

    dev_meta = load_data(SPLITS_DIR / "train_split.csv", SPLITS_DIR / "val_split.csv", SPLITS_DIR / "test_split.csv")
    dev_meta = pd.concat([train_meta, val_meta], ignore_index=True)
    dev_meta["patient_id"] = dev_meta["patient_id"].astype(str)
    dev_meta["label"] = dev_meta["label"].astype(int)

    patient_to_outer_fold = (
        create_outer_patient_folds(
            dev_meta
        )
    )

    dev_patient_ids = set(
        dev_meta[
            "patient_id"
        ].unique()
    )

    (
        local_dev,
        global_dev,
    ) = load_preprocessed_indexes(
        dev_patient_ids
    )

    if args.all:
        folds = range(N_OUTER_FOLDS)
    else:
        folds = [
            args.fold
        ]

    for fold in folds:
        run_one_fold(
            outer_fold=fold,
            dev_meta=dev_meta,
            patient_to_outer_fold=
                patient_to_outer_fold,
            local_dev=local_dev,
            global_dev=global_dev,
            force=args.force,
        )


if __name__ == "__main__":
    main()