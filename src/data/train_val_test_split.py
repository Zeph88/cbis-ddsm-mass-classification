import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

from src.config import SEED, DATASET_INDEX, SPLITS_DIR

VAL_PRC = 0.20
N_SPLIT_TRIALS = 50

df = pd.read_csv(DATASET_INDEX)
df["label"] = df["label"].astype(int)

def print_split_summary(
    split_name,
    split_df,
):
    """
    Prints the number of images, patients and the malignant-class
    prevalence for one split.
    """

    n_images = len(split_df)
    n_patients = split_df["patient_id"].nunique()
    malignant_count = int(split_df["label"].sum())
    malignant_rate = float(split_df["label"].mean())

    print("\n" + "=" * 60)
    print(split_name)
    print("=" * 60)

    print(f"Images:             {n_images}")
    print(f"Patients:           {n_patients}")
    print(f"Benign images:      {n_images - malignant_count}")
    print(f"Malignant images:   {malignant_count}")
    print(f"Malignant rate:     {malignant_rate:.4%}")


def check_no_patient_leakage(
    train_df,
    val_df,
    test_df=None,
):
    """
    Checks that no patient occurs in more than one split.
    """

    train_patients = set(
        train_df["patient_id"].dropna()
    )

    val_patients = set(
        val_df["patient_id"].dropna()
    )

    train_val_overlap = (
        train_patients & val_patients
    )

    train_test_overlap = set()
    val_test_overlap = set()

    print("\nPatient leakage checks")
    print("-" * 60)

    print(
        "train ∩ val patients:",
        len(train_val_overlap),
    )

    if train_val_overlap:
        raise ValueError(
            "Patient leakage between train and validation: "
            f"{sorted(train_val_overlap)}"
        )

    if test_df is not None:
        test_patients = set(
            test_df["patient_id"].dropna()
        )

        train_test_overlap = (
            train_patients & test_patients
        )

        val_test_overlap = (
            val_patients & test_patients
        )

        print(
            "train ∩ test patients:",
            len(train_test_overlap),
        )

        print(
            "val ∩ test patients:",
            len(val_test_overlap),
        )

        if train_test_overlap:
            raise ValueError(
                "Patient leakage between train and test: "
                f"{sorted(train_test_overlap)}"
            )

        if val_test_overlap:
            raise ValueError(
                "Patient leakage between validation and test: "
                f"{sorted(val_test_overlap)}"
            )

    return (
        len(train_val_overlap),
        len(train_test_overlap),
        len(val_test_overlap),
    )


# ================================================================
# Stratified grouped split
# ================================================================

def train_val_split_stratified_no_leakage(
    development_df,
    val_prc=0.20,
    seed=SEED,
    n_trials=50,
):
    """
    Splits the official training set into development-train and
    validation subsets.

    Objectives:
    - no patient leakage;
    - validation size close to val_prc;
    - train and validation malignant rates close to the malignant
      rate of the complete official training set.

    StratifiedGroupKFold preserves groups and attempts to preserve
    the sample-level class distribution.

    Several deterministic seeds are tested. Selection is based only
    on split size and class prevalence, never on model performance.
    """

    if not 0.0 < val_prc < 1.0:
        raise ValueError(
            "val_prc must be between 0 and 1."
        )

    development_df = (
        development_df
        .copy()
        .reset_index(drop=True)
    )

    if development_df.empty:
        raise ValueError(
            "The official training dataframe is empty."
        )

    n_unique_patients = (
        development_df["patient_id"].nunique()
    )

    # For val_prc=0.20, this produces five folds.
    n_splits = int(round(1.0 / val_prc))

    if n_splits < 2:
        raise ValueError(
            "val_prc produces fewer than two folds."
        )

    if n_unique_patients < n_splits:
        raise ValueError(
            f"Only {n_unique_patients} patients are available, "
            f"but {n_splits} folds are required."
        )

    effective_val_prc = 1.0 / n_splits

    if not np.isclose(
        effective_val_prc,
        val_prc,
        atol=0.02,
    ):
        raise ValueError(
            "This implementation requires a validation proportion "
            "close to 1 / n_splits. For example, use 0.20 for "
            "five folds."
        )

    overall_malignant_rate = float(
        development_df["label"].mean()
    )

    labels = (
        development_df["label"]
        .to_numpy()
    )

    groups = (
        development_df["patient_id"]
        .to_numpy()
    )

    # X is ignored by the splitter, but it must have the right
    # number of observations.
    dummy_x = np.zeros(
        shape=(len(development_df), 1),
        dtype=np.float32,
    )

    best_candidate = None

    for trial in range(n_trials):
        trial_seed = seed + trial

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=trial_seed,
        )

        for fold_number, (
            train_indices,
            val_indices,
        ) in enumerate(
            splitter.split(
                dummy_x,
                labels,
                groups,
            )
        ):
            candidate_train = (
                development_df
                .iloc[train_indices]
            )

            candidate_val = (
                development_df
                .iloc[val_indices]
            )

            train_rate = float(
                candidate_train["label"].mean()
            )

            val_rate = float(
                candidate_val["label"].mean()
            )

            actual_val_prc = (
                len(candidate_val)
                / len(development_df)
            )

            train_rate_error = abs(
                train_rate
                - overall_malignant_rate
            )

            val_rate_error = abs(
                val_rate
                - overall_malignant_rate
            )

            size_error = abs(
                actual_val_prc
                - val_prc
            )

            # The maximum prevalence deviation prevents a candidate
            # with one excellent split and one poor split from being
            # selected.
            prevalence_error = max(
                train_rate_error,
                val_rate_error,
            )

            score = (
                prevalence_error
                + size_error
            )

            candidate = {
                "score": score,
                "trial_seed": trial_seed,
                "fold_number": fold_number,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_rate": train_rate,
                "val_rate": val_rate,
                "actual_val_prc": actual_val_prc,
                "prevalence_error": prevalence_error,
                "size_error": size_error,
            }

            if (
                best_candidate is None
                or candidate["score"]
                < best_candidate["score"]
            ):
                best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            "No valid train/validation split was generated."
        )

    train_df = (
        development_df
        .iloc[best_candidate["train_indices"]]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        development_df
        .iloc[best_candidate["val_indices"]]
        .copy()
        .reset_index(drop=True)
    )

    print("\nSelected stratified grouped split")
    print("-" * 60)

    print(
        "Official-train malignant rate: "
        f"{overall_malignant_rate:.4%}"
    )

    print(
        "New train malignant rate:      "
        f"{best_candidate['train_rate']:.4%}"
    )

    print(
        "New validation malignant rate: "
        f"{best_candidate['val_rate']:.4%}"
    )

    print(
        "Validation proportion:         "
        f"{best_candidate['actual_val_prc']:.4%}"
    )

    print(
        "Selected random seed:          "
        f"{best_candidate['trial_seed']}"
    )

    print(
        "Selected fold:                 "
        f"{best_candidate['fold_number']}"
    )

    print(
        "Prevalence deviation:          "
        f"{best_candidate['prevalence_error']:.6f}"
    )

    print(
        "Validation-size deviation:     "
        f"{best_candidate['size_error']:.6f}"
    )

    return train_df, val_df


# ================================================================
# Preserve the official train/test partition
# ================================================================

official_train_df = df[
    (df["keep"] == True)
    & (df["source"] == "train")
].copy().reset_index(drop=True)

test_df = df[
    (df["keep"] == True)
    & (df["source"] == "test")
].copy().reset_index(drop=True)


print_split_summary(
    "OFFICIAL TRAIN BEFORE TRAIN/VAL SPLIT",
    official_train_df,
)

print_split_summary(
    "OFFICIAL TEST — UNCHANGED",
    test_df,
)


# ================================================================
# Create the new train/validation split
# ================================================================

train_df, val_df = (
    train_val_split_stratified_no_leakage(
        development_df=official_train_df,
        val_prc=VAL_PRC,
        seed=SEED,
        n_trials=N_SPLIT_TRIALS,
    )
)


# ================================================================
# Final diagnostics
# ================================================================

print_split_summary(
    "NEW DEVELOPMENT TRAIN",
    train_df,
)

print_split_summary(
    "NEW VALIDATION",
    val_df,
)

print_split_summary(
    "OFFICIAL TEST — UNCHANGED",
    test_df,
)


overlap_counts = check_no_patient_leakage(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
)

train_val_overlap = overlap_counts[0]
train_test_overlap = overlap_counts[1]
val_test_overlap = overlap_counts[2]


# Verify that all official-train rows were assigned exactly once.
expected_train_rows = len(official_train_df)

actual_train_val_rows = (
    len(train_df)
    + len(val_df)
)

if actual_train_val_rows != expected_train_rows:
    raise ValueError(
        "Some official-training rows were lost or duplicated: "
        f"expected {expected_train_rows}, "
        f"found {actual_train_val_rows}."
    )


# ================================================================
# Save only after all checks pass
# ================================================================

if (
    train_val_overlap == 0
    and train_test_overlap == 0
    and val_test_overlap == 0
):
    train_path = (
        SPLITS_DIR / "train_split.csv"
    )

    val_path = (
        SPLITS_DIR / "val_split.csv"
    )

    test_path = (
        SPLITS_DIR / "test_split.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    val_df.to_csv(
        val_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print("\nSplits successfully saved")
    print("-" * 60)
    print(f"Train:      {train_path}")
    print(f"Validation: {val_path}")
    print(f"Test:       {test_path}")