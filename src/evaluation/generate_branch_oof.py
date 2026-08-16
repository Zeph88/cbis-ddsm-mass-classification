import argparse
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.config import (
    BATCH_SIZE,
    GLOBAL_HEIGHT,
    GLOBAL_WIDTH,
    LOCAL_HEIGHT,
    LOCAL_WIDTH,
    OUTPUT_MODEL,
    OUTPUT_NPY,
    SEED,
)

from src.training.dataset_preparation import (
    build_tf_dataset,
)


# ======================================================================
# Configuration
# ======================================================================

N_OUTER_FOLDS = 5
TRAINING_SEED = SEED

CV_ROOT = (
    OUTPUT_MODEL
    / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{TRAINING_SEED}"
)

FOLDS_DIR = CV_ROOT / "folds"

MAMMOGRAM_KEY = [
    "patient_id",
    "left or right breast",
    "image view",
]


# ======================================================================
# Metrics
# ======================================================================

def calculate_metrics(
    y_true,
    probability,
):
    return {
        "auc": roc_auc_score(
            y_true,
            probability,
        ),
        "bce": log_loss(
            y_true,
            probability,
            labels=[0, 1],
        ),
        "ap": average_precision_score(
            y_true,
            probability,
        ),
        "brier": brier_score_loss(
            y_true,
            probability,
        ),
    }


# ======================================================================
# Load preprocessed indexes
# ======================================================================

def load_preprocessed_indexes():
    local_path = (
        OUTPUT_NPY
        / f"dataset_index_zoom_{LOCAL_HEIGHT}x{LOCAL_WIDTH}.csv"
    )

    global_path = (
        OUTPUT_NPY
        / f"dataset_index_full_{GLOBAL_HEIGHT}x{GLOBAL_WIDTH}.csv"
    )

    if not local_path.exists():
        raise FileNotFoundError(
            f"Missing local index: {local_path}"
        )

    if not global_path.exists():
        raise FileNotFoundError(
            f"Missing global index: {global_path}"
        )

    local_df = pd.read_csv(
        local_path
    )

    global_df = pd.read_csv(
        global_path
    )

    # Match the CV representation of patient IDs.
    local_df["patient_id"] = (
        local_df["patient_id"]
        .astype(str)
    )

    global_df["patient_id"] = (
        global_df["patient_id"]
        .astype(str)
    )

    if "label" in local_df.columns:
        local_df["label"] = (
            local_df["label"]
            .astype(int)
        )

    return (
        local_df,
        global_df,
    )


# ======================================================================
# Recover the exact outer-evaluation observations
# ======================================================================

def build_outer_dataframe(
    fold,
    local_index,
    global_index,
):
    fold_dir = (
        FOLDS_DIR
        / f"fold_{fold}"
    )

    oof_path = (
        fold_dir
        / "oof_predictions.csv"
    )

    if not oof_path.exists():
        raise FileNotFoundError(
            f"Missing fusion OOF file: {oof_path}"
        )

    reference_df = pd.read_csv(
        oof_path
    )

    required_reference_columns = {
        "sample_id",
        "patient_id",
        "left or right breast",
        "image view",
        "label",
        "fold",
    }

    missing = (
        required_reference_columns
        - set(reference_df.columns)
    )

    if missing:
        raise ValueError(
            "OOF file is missing required columns: "
            f"{sorted(missing)}"
        )

    reference_df["patient_id"] = (
        reference_df["patient_id"]
        .astype(str)
    )

    reference_df["label"] = (
        reference_df["label"]
        .astype(int)
    )

    reference_df["fold"] = (
        reference_df["fold"]
        .astype(int)
    )

    if not (
        reference_df["fold"] == fold
    ).all():
        raise ValueError(
            f"OOF file for fold {fold} contains "
            "rows assigned to another fold."
        )

    if reference_df[
        "sample_id"
    ].duplicated().any():
        raise ValueError(
            f"Duplicate sample_id values in fold {fold} "
            "OOF predictions."
        )

    # Preserve the exact order already used for fusion OOF predictions.
    reference_df["_row_order"] = np.arange(
        len(reference_df)
    )

    # Use string only as a robust merge key.
    reference_df["_sample_key"] = (
        reference_df["sample_id"]
        .astype(str)
    )

    # --------------------------------------------------------------
    # LOCAL IMAGE PATH
    # --------------------------------------------------------------

    local_lookup = (
        local_index.copy()
    )

    local_lookup["_sample_key"] = (
        local_lookup["sample_id"]
        .astype(str)
    )

    local_lookup["patient_id"] = (
        local_lookup["patient_id"]
        .astype(str)
    )

    local_lookup["label"] = (
        local_lookup["label"]
        .astype(int)
    )

    # Restrict lookup to patients present in this outer fold.
    outer_patients = set(
        reference_df["patient_id"]
    )

    local_lookup = local_lookup[
        local_lookup["patient_id"].isin(
            outer_patients
        )
    ].copy()

    local_merge_keys = [
        "_sample_key",
        "patient_id",
        "left or right breast",
        "image view",
        "label",
    ]

    if local_lookup[
        "_sample_key"
    ].duplicated().any():
        duplicated = local_lookup.loc[
            local_lookup[
                "_sample_key"
            ].duplicated(
                keep=False
            ),
            [
                "sample_id",
                "patient_id",
                "left or right breast",
                "image view",
            ],
        ]

        raise ValueError(
            "Duplicate local sample IDs found:\n"
            f"{duplicated.head()}"
        )

    paired_df = reference_df.merge(
        local_lookup[
            local_merge_keys
            + [
                "preprocessed_image_path",
            ]
        ],
        on=local_merge_keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    paired_df = paired_df.rename(
        columns={
            "preprocessed_image_path":
                "local_path",
        }
    )

    if paired_df[
        "local_path"
    ].isna().any():
        missing_local = paired_df.loc[
            paired_df["local_path"].isna(),
            [
                "sample_id",
                "patient_id",
                "left or right breast",
                "image view",
                "label",
            ],
        ]

        raise RuntimeError(
            "Could not recover local image paths for "
            "some OOF observations:\n"
            f"{missing_local.head()}"
        )

    # --------------------------------------------------------------
    # GLOBAL IMAGE PATH
    #
    # Reproduce the same many-to-one pairing rule as the CV script:
    # one full mammogram per
    # patient_id + laterality + view.
    # --------------------------------------------------------------

    global_lookup_source = (
        global_index[
            global_index["patient_id"].isin(
                outer_patients
            )
        ]
        .copy()
    )

    global_path_count = (
        global_lookup_source
        .groupby(
            MAMMOGRAM_KEY
        )[
            "preprocessed_image_path"
        ]
        .nunique()
    )

    conflicting_global_paths = (
        global_path_count[
            global_path_count > 1
        ]
    )

    if not conflicting_global_paths.empty:
        raise RuntimeError(
            "A mammogram key refers to several "
            "global image paths:\n"
            f"{conflicting_global_paths.head()}"
        )

    global_lookup = (
        global_lookup_source[
            MAMMOGRAM_KEY
            + [
                "preprocessed_image_path",
            ]
        ]
        .drop_duplicates(
            subset=MAMMOGRAM_KEY
        )
        .rename(
            columns={
                "preprocessed_image_path":
                    "global_path",
            }
        )
    )

    paired_df = paired_df.merge(
        global_lookup,
        on=MAMMOGRAM_KEY,
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if paired_df[
        "global_path"
    ].isna().any():
        missing_global = paired_df.loc[
            paired_df["global_path"].isna(),
            [
                "sample_id",
                "patient_id",
                "left or right breast",
                "image view",
                "label",
            ],
        ]

        raise RuntimeError(
            "Could not recover global image paths for "
            "some OOF observations:\n"
            f"{missing_global.head()}"
        )

    # Restore exact fusion OOF row order.
    paired_df = (
        paired_df
        .sort_values(
            "_row_order"
        )
        .reset_index(
            drop=True
        )
    )

    if len(paired_df) != len(reference_df):
        raise RuntimeError(
            "Outer-data reconstruction changed the "
            "number of observations."
        )

    if not np.array_equal(
        paired_df["sample_id"].to_numpy(),
        reference_df["sample_id"].to_numpy(),
    ):
        raise RuntimeError(
            "Outer-data sample order does not match "
            "the existing fusion OOF file."
        )

    if not np.array_equal(
        paired_df["label"].to_numpy(),
        reference_df["label"].to_numpy(),
    ):
        raise RuntimeError(
            "Outer-data labels do not match the "
            "existing fusion OOF file."
        )

    return paired_df


# ======================================================================
# Build branch inference datasets
# ======================================================================

def build_local_outer_dataset(
    outer_df,
):
    return build_tf_dataset(
        outer_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=TRAINING_SEED,
        path_image="local_path",
        image_height=LOCAL_HEIGHT,
        image_width=LOCAL_WIDTH,
    )


def build_global_outer_dataset(
    outer_df,
):
    return build_tf_dataset(
        outer_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=TRAINING_SEED,
        path_image="global_path",
        image_height=GLOBAL_HEIGHT,
        image_width=GLOBAL_WIDTH,
    )


# ======================================================================
# Predict one saved branch
# ======================================================================

def predict_saved_model(
    model_path,
    dataset,
    expected_length,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {model_path}"
        )

    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = (
        model.predict(
            dataset,
            verbose=1,
        )
        .reshape(-1)
    )

    if len(probabilities) != expected_length:
        raise RuntimeError(
            "Prediction/metadata mismatch: "
            f"{len(probabilities)} predictions for "
            f"{expected_length} observations."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            f"Non-finite predictions produced by "
            f"{model_path}."
        )

    if (
        (probabilities < 0).any()
        or
        (probabilities > 1).any()
    ):
        raise RuntimeError(
            f"Probabilities outside [0, 1] produced "
            f"by {model_path}."
        )

    del model

    tf.keras.backend.clear_session()
    gc.collect()

    return probabilities


# ======================================================================
# Generate one fold
# ======================================================================

def generate_fold_predictions(
    fold,
    local_index,
    global_index,
    overwrite=False,
):
    fold_dir = (
        FOLDS_DIR
        / f"fold_{fold}"
    )

    output_path = (
        fold_dir
        / "branch_oof_predictions.csv"
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        print(
            f"Fold {fold}: already exists, skipping:"
        )
        print(
            output_path
        )
        return

    local_model_path = (
        fold_dir
        / "local.keras"
    )

    global_model_path = (
        fold_dir
        / "global.keras"
    )

    print()
    print(
        "=" * 70
    )
    print(
        f"FOLD {fold}"
    )
    print(
        "=" * 70
    )

    outer_df = build_outer_dataframe(
        fold=fold,
        local_index=local_index,
        global_index=global_index,
    )

    print(
        "Outer observations:",
        len(outer_df),
    )

    print(
        "Outer patients:",
        outer_df[
            "patient_id"
        ].nunique(),
    )

    print(
        "Malignant prevalence:",
        float(
            outer_df[
                "label"
            ].mean()
        ),
    )

    # --------------------------------------------------------------
    # LOCAL
    # --------------------------------------------------------------

    print(
        "\nPredicting saved LOCAL checkpoint..."
    )

    local_ds = (
        build_local_outer_dataset(
            outer_df
        )
    )

    local_probability = (
        predict_saved_model(
            model_path=local_model_path,
            dataset=local_ds,
            expected_length=len(
                outer_df
            ),
        )
    )

    del local_ds
    gc.collect()

    # --------------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------------

    print(
        "\nPredicting saved GLOBAL checkpoint..."
    )

    global_ds = (
        build_global_outer_dataset(
            outer_df
        )
    )

    global_probability = (
        predict_saved_model(
            model_path=global_model_path,
            dataset=global_ds,
            expected_length=len(
                outer_df
            ),
        )
    )

    del global_ds
    gc.collect()

    # --------------------------------------------------------------
    # Create exactly the file expected by analyse_oof.py
    # --------------------------------------------------------------

    branch_oof = outer_df[
        [
            "sample_id",
            "patient_id",
            "fold",
            "label",
        ]
    ].copy()

    branch_oof[
        "local_probability"
    ] = local_probability

    branch_oof[
        "global_probability"
    ] = global_probability

    # Final integrity checks before writing.
    if branch_oof[
        "sample_id"
    ].duplicated().any():
        raise RuntimeError(
            f"Duplicate sample IDs in fold {fold}."
        )

    if branch_oof[
        [
            "local_probability",
            "global_probability",
        ]
    ].isna().any().any():
        raise RuntimeError(
            f"Missing branch predictions in fold {fold}."
        )

    branch_oof.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Optional immediate sanity-check metrics
    # --------------------------------------------------------------

    y_true = (
        branch_oof[
            "label"
        ]
        .astype(int)
        .to_numpy()
    )

    local_scores = (
        calculate_metrics(
            y_true,
            local_probability,
        )
    )

    global_scores = (
        calculate_metrics(
            y_true,
            global_probability,
        )
    )

    print(
        "\nLocal:",
        local_scores,
    )

    print(
        "Global:",
        global_scores,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate local/global OOF probabilities "
            "from already-trained CV checkpoints. "
            "This script performs inference only."
        )
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--fold",
        type=int,
        choices=range(
            N_OUTER_FOLDS
        ),
        help=(
            "Generate branch OOF predictions "
            "for one outer fold."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Generate branch OOF predictions "
            "for all outer folds."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite an existing "
            "branch_oof_predictions.csv."
        ),
    )

    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    local_index, global_index = (
        load_preprocessed_indexes()
    )

    if args.all:
        folds = range(
            N_OUTER_FOLDS
        )
    else:
        folds = [
            args.fold
        ]

    for fold in folds:
        generate_fold_predictions(
            fold=fold,
            local_index=local_index,
            global_index=global_index,
            overwrite=args.overwrite,
        )

    print()
    print(
        "=" * 70
    )
    print(
        "Branch OOF generation complete."
    )
    print(
        "No models were trained."
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()