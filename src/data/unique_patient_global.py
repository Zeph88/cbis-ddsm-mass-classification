from pathlib import Path
import pandas as pd
from src.config import OUTPUT_NPY


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MAMMOGRAM_COLUMNS = [
    "patient_id",
    "left or right breast",
    "image view",
]

LABEL_COLUMN = "label"

SPLIT_FILES = {
    "train": {
        "input": "train_split.csv",
        "output": "train_split_global.csv",
    },
    "val": {
        "input": "val_split.csv",
        "output": "val_split_global.csv",
    },
    "test": {
        "input": "test_split.csv",
        "output": "test_split_global.csv",
    },
}


# ------------------------------------------------------------------
# Mammogram-level dataset creation
# ------------------------------------------------------------------

def create_global_split(
    input_path: Path,
    output_path: Path,
    mammogram_columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    """
    Convert a lesion-level split into a mammogram-level split.

    One row is retained per mammogram. When the same mammogram is
    associated with both benign and malignant lesions, the malignant
    row is retained.
    """

    df = pd.read_csv(input_path)

    required_columns = set(
        mammogram_columns + [label_column]
    )

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in {input_path.name}: "
            f"{sorted(missing_columns)}"
        )

    if df[label_column].isna().any():
        raise ValueError(
            f"{label_column} contains missing values "
            f"in {input_path.name}."
        )

    observed_labels = set(
        df[label_column].astype(int).unique()
    )

    if not observed_labels.issubset({0, 1}):
        raise ValueError(
            f"{label_column} must contain only 0 and 1. "
            f"Observed values in {input_path.name}: "
            f"{sorted(observed_labels)}"
        )

    df = df.copy()
    df[label_column] = df[label_column].astype(int)

    # Count the number of lesion rows associated with each mammogram.
    mammogram_sizes = (
        df.groupby(
            mammogram_columns,
            dropna=False,
        )
        .size()
    )

    multi_lesion_count = int(
        (mammogram_sizes > 1).sum()
    )

    # Identify mammograms containing both benign and malignant lesions.
    distinct_label_counts = (
        df.groupby(
            mammogram_columns,
            dropna=False,
        )[label_column]
        .nunique()
    )

    mixed_label_count = int(
        (distinct_label_counts > 1).sum()
    )

    # Calculate the expected mammogram-level label.
    # A mammogram is malignant when at least one lesion is malignant.
    expected_labels = (
        df.groupby(
            mammogram_columns,
            dropna=False,
        )[label_column]
        .max()
        .sort_index()
    )

    # Sort malignant rows first, then keep one representative row
    # for each mammogram.
    global_df = (
        df.sort_values(
            by=label_column,
            ascending=False,
            kind="stable",
        )
        .drop_duplicates(
            subset=mammogram_columns,
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Verify that the retained labels match the expected maximum labels.
    retained_labels = (
        global_df
        .set_index(mammogram_columns)[label_column]
        .sort_index()
    )

    pd.testing.assert_series_equal(
        retained_labels,
        expected_labels,
        check_names=False,
    )

    # Verify that only one row remains per mammogram.
    remaining_duplicates = global_df.duplicated(
        subset=mammogram_columns,
        keep=False,
    )

    if remaining_duplicates.any():
        raise RuntimeError(
            f"Duplicate mammograms remain in {output_path.name}."
        )

    global_df.to_csv(
        output_path,
        index=False,
    )

    malignant_count = int(
        global_df[label_column].sum()
    )

    total_count = len(global_df)
    benign_count = total_count - malignant_count
    malignant_ratio = global_df[label_column].mean()

    print("\n" + "=" * 70)
    print(f"Input:  {input_path.name}")
    print(f"Output: {output_path.name}")
    print("=" * 70)
    print(f"Lesion-level rows:             {len(df)}")
    print(f"Unique mammograms:             {total_count}")
    print(f"Rows removed:                  {len(df) - total_count}")
    print(f"Multi-lesion mammograms:       {multi_lesion_count}")
    print(f"Mixed-label mammograms:        {mixed_label_count}")
    print(f"Benign mammograms:             {benign_count}")
    print(f"Malignant mammograms:          {malignant_count}")
    print(f"Malignant ratio:               {malignant_ratio:.3%}")

    return global_df


# ------------------------------------------------------------------
# Generate the three global split files
# ------------------------------------------------------------------

global_splits = {}

for split_name, filenames in SPLIT_FILES.items():
    input_path = (
        OUTPUT_NPY
        / filenames["input"]
    )

    output_path = (
        OUTPUT_NPY
        / filenames["output"]
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    global_splits[split_name] = create_global_split(
        input_path=input_path,
        output_path=output_path,
        mammogram_columns=MAMMOGRAM_COLUMNS,
        label_column=LABEL_COLUMN,
    )


# ------------------------------------------------------------------
# Cross-split controls
# ------------------------------------------------------------------

combined_global_df = pd.concat(
    [
        split_df.assign(source_split=split_name)
        for split_name, split_df
        in global_splits.items()
    ],
    ignore_index=True,
)

# Verify that no patient appears in multiple splits.
patient_split_counts = (
    combined_global_df
    .groupby("patient_id")["source_split"]
    .nunique()
)

leaking_patients = patient_split_counts[
    patient_split_counts > 1
]

if not leaking_patients.empty:
    raise ValueError(
        f"{len(leaking_patients)} patients appear "
        "in multiple global splits."
    )

print("\nNo patient leakage detected across global splits.")


# ------------------------------------------------------------------
# Compare class prevalence across splits
# ------------------------------------------------------------------

split_summary = pd.DataFrame(
    {
        split_name: {
            "mammograms": len(split_df),
            "benign": int(
                (split_df[LABEL_COLUMN] == 0).sum()
            ),
            "malignant": int(
                (split_df[LABEL_COLUMN] == 1).sum()
            ),
            "malignant_ratio": float(
                split_df[LABEL_COLUMN].mean()
            ),
        }
        for split_name, split_df
        in global_splits.items()
    }
).T

print("\n" + "=" * 70)
print("GLOBAL SPLIT DISTRIBUTION")
print("=" * 70)
print(split_summary)


train_ratio = split_summary.loc[
    "train",
    "malignant_ratio",
]

val_ratio = split_summary.loc[
    "val",
    "malignant_ratio",
]

train_val_difference = abs(
    train_ratio - val_ratio
)

print(
    f"\nTrain malignant ratio:      "
    f"{train_ratio:.3%}"
)

print(
    f"Validation malignant ratio: "
    f"{val_ratio:.3%}"
)

print(
    f"Absolute difference:        "
    f"{train_val_difference:.3%}"
)


# Perfect equality may be impossible because patients remain grouped.
MAX_ALLOWED_DIFFERENCE = 0.02

if train_val_difference > MAX_ALLOWED_DIFFERENCE:
    raise ValueError(
        "Train and validation malignant ratios differ by "
        f"{train_val_difference:.3%}, which exceeds the "
        f"allowed difference of "
        f"{MAX_ALLOWED_DIFFERENCE:.1%}."
    )

print(
    "\nTrain and validation malignant ratios "
    "are sufficiently similar."
)