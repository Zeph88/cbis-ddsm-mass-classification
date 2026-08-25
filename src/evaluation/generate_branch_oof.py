import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from src.functions import load_data, parse_arguments
from src.config import BATCH_SIZE, GLOBAL_HEIGHT, GLOBAL_WIDTH, LOCAL_HEIGHT, LOCAL_WIDTH, OUTPUT_MODEL, OUTPUT_NPY, SEED, N_OUTER_FOLDS, MAMMOGRAM_KEY
from src.training.dataset_preparation import build_tf_dataset
from src.evaluation.evaluation_utils import calculate_metrics, collect_binary_predictions, load_preprocessed_indexes
from src.data.pairing import build_global_lookup

CV_ROOT = (OUTPUT_MODEL / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{SEED}")
FOLDS_DIR = CV_ROOT / "folds"

# Recover the exact outer-evaluation observations
def build_outer_dataframe(fold, local_index, global_index):
    
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    oof_path = fold_dir / "oof_predictions.csv"

    if not oof_path.exists():
        raise FileNotFoundError(f"Missing fusion OOF file: {oof_path}")

    reference_df = pd.read_csv(oof_path)

    required_reference_columns = {
        "sample_id",
        "patient_id",
        "left or right breast",
        "image view",
        "label",
        "fold",
    }

    missing = (required_reference_columns - set(reference_df.columns))

    if missing:
        raise ValueError(f"OOF file is missing required columns: {sorted(missing)}")

    reference_df["patient_id"] = (reference_df["patient_id"].astype(str))
    reference_df["label"] = (reference_df["label"].astype(int))
    reference_df["fold"] = (reference_df["fold"].astype(int))

    if not (reference_df["fold"] == fold).all():
        raise ValueError(f"OOF file for fold {fold} contains rows assigned to another fold.")

    if reference_df["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values in fold {fold} OOF predictions.")

    # Preserve the exact order already used for fusion OOF predictions.
    reference_df["_row_order"] = np.arange(len(reference_df))

    # Use string only as a robust merge key.
    reference_df["_sample_key"] = (reference_df["sample_id"].astype(str))

    # LOCAL IMAGE PATH
    local_lookup = (local_index.copy())
    local_lookup["_sample_key"] = (local_lookup["sample_id"].astype(str))
    local_lookup["patient_id"] = (local_lookup["patient_id"].astype(str))
    local_lookup["label"] = (local_lookup["label"].astype(int))

    # Restrict lookup to patients present in this outer fold.
    outer_patients = set(reference_df["patient_id"])

    local_lookup = local_lookup[local_lookup["patient_id"].isin(outer_patients)].copy()

    # local_merge_keys = [
    #     "_sample_key",
    #     "patient_id",
    #     "left or right breast",
    #     "image view",
    #     "label",
    # ]

    if local_lookup["_sample_key"].duplicated().any():
        duplicated = local_lookup.loc[local_lookup["_sample_key"].duplicated(keep=False), ["sample_id", "patient_id", "left or right breast", "image view"]]
        raise ValueError(f"Duplicate local sample IDs found: {duplicated.head()}")

    paired_df = reference_df.merge(
        local_lookup[["_sample_key", "preprocessed_image_path"]],
        on="_sample_key", how="left", validate="one_to_one", sort=False
    )

    paired_df = paired_df.rename(columns={"preprocessed_image_path": "local_path"})

    if paired_df["local_path"].isna().any():
        missing_local = paired_df.loc[paired_df["local_path"].isna(), ["sample_id", "patient_id", "left or right breast", "image view", "label"]]
        raise RuntimeError(f"Could not recover local image paths for some OOF observations: {missing_local.head()}")

    # GLOBAL IMAGE PATH
    global_lookup_source = (global_index[global_index["patient_id"].isin(outer_patients)].copy())
    global_lookup = build_global_lookup(global_lookup_source)
    
    paired_df = paired_df.merge(global_lookup, on=MAMMOGRAM_KEY, how="left", validate="many_to_one", sort=False)

    if paired_df["global_path"].isna().any():
        missing_global = paired_df.loc[paired_df["global_path"].isna(), ["sample_id"]]
        raise RuntimeError(f"Could not recover global image paths for some OOF observations: {missing_global.head()}")

    # Restore exact fusion OOF row order.
    paired_df = (paired_df.sort_values("_row_order").reset_index(drop=True))

    if len(paired_df) != len(reference_df):
        raise RuntimeError("Outer-data reconstruction changed the number of observations.")

    if not np.array_equal(paired_df["sample_id"].to_numpy(), reference_df["sample_id"].to_numpy()):
        raise RuntimeError("Outer-data sample order does not match the existing fusion OOF file.")

    if not np.array_equal(paired_df["label"].to_numpy(), reference_df["label"].to_numpy()):
        raise RuntimeError("Outer-data labels do not match the existing fusion OOF file.")

    return paired_df


# Build branch inference datasets
def build_local_outer_dataset(outer_df):
    return build_tf_dataset(
        outer_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        path_image="local_path",
        image_height=LOCAL_HEIGHT,
        image_width=LOCAL_WIDTH,
    )


def build_global_outer_dataset(outer_df):
    return build_tf_dataset(
        outer_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        path_image="global_path",
        image_height=GLOBAL_HEIGHT,
        image_width=GLOBAL_WIDTH,
    )


def collect_saved_model_predictions(model_path, dataset):
    tf.keras.backend.clear_session()
    gc.collect()

    model = tf.keras.models.load_model(model_path, compile=False)
    y_true, probabilities = collect_binary_predictions(model, dataset)

    del model
    del dataset
    tf.keras.backend.clear_session()
    gc.collect()

    return y_true, probabilities

# Generate one fold
def generate_fold_predictions(fold, local_index, global_index, overwrite=False):
    
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    output_path = fold_dir / "branch_oof_predictions.csv"

    if output_path.exists() and not overwrite:
        print(f"Fold {fold}: already exists, skipping:")
        print(output_path)
        return

    local_model_path = fold_dir / "local.keras"
    global_model_path = fold_dir / "global.keras"

    print(f"FOLD {fold}")

    outer_df = build_outer_dataframe(fold=fold, local_index=local_index, global_index=global_index)

    print("Outer observations:", len(outer_df))
    print("Outer patients:", outer_df["patient_id"].nunique())

    print("Malignant prevalence:", float(outer_df["label"].mean()))

    # LOCAL
    print("Predicting saved LOCAL checkpoint...")

    local_ds = build_local_outer_dataset(outer_df)
    local_true, local_probability = collect_saved_model_predictions(model_path=local_model_path, dataset=local_ds)

    # GLOBAL
    print("\nPredicting saved GLOBAL checkpoint...")


    global_ds = build_global_outer_dataset(outer_df)
    global_true, global_probability = collect_saved_model_predictions(model_path=global_model_path, dataset=global_ds)


    y_true = outer_df["label"].astype(np.int32).to_numpy()

    # sanity-check metrics
    if not np.array_equal(local_true, y_true):
        raise RuntimeError("Local dataset labels do not match outer metadata.")

    if not np.array_equal(global_true, y_true):
        raise RuntimeError("Global dataset labels do not match outer metadata.")

    # Create exactly the file expected by analyse_oof.py
    branch_oof = outer_df[["sample_id", "fold"]].copy()

    branch_oof["local_probability"] = local_probability
    branch_oof["global_probability"] = global_probability

    if branch_oof["sample_id"].duplicated().any():
        raise RuntimeError(f"Duplicate sample IDs in fold {fold}.")

    if branch_oof[["local_probability", "global_probability"]].isna().any().any():
        raise RuntimeError(f"Missing branch predictions in fold {fold}.")

    branch_oof.to_csv(output_path, index=False)

    local_scores = calculate_metrics(y_true, local_probability)
    global_scores = calculate_metrics(y_true, global_probability)

    print("\nLocal:", local_scores)
    print("Global:", global_scores)
    print(f"Saved to {output_path}")

def main():

    args = parse_arguments(
        description=("Generate local/global OOF probabilities from already-trained CV checkpoints."),
        arguments=[
            {
                "name": "--overwrite",
                "action": "store_true",
                "help": ("Overwrite an existing branch_oof_predictions.csv.")
            }
        ],
        exclusive_arguments=[
            {
                "name": "--fold",
                "type": int,
                "choices": range(N_OUTER_FOLDS),
                "help": ("Generate branch OOF predictions for one outer fold.")
            },
            {
                "name": "--all",
                "action": "store_true",
                "help": ("Generate branch OOF predictions for all outer folds.")
            }
        ],
        exclusive_required=True,
    )
    local_index, global_index = (load_preprocessed_indexes())

    if args.all:
        folds = range(N_OUTER_FOLDS)
    else:
        folds = [args.fold]

    for fold in folds:
        generate_fold_predictions(
            fold=fold,
            local_index=local_index,
            global_index=global_index,
            overwrite=args.overwrite,
        )

    print("Branch OOF generation complete.")

if __name__ == "__main__":
    main()