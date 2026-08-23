from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUT_PLOT, SPLITS_DIR
from src.functions import ensure_directory

ensure_directory(OUTPUT_PLOT)

SPLIT_FILES = {
    "Train": SPLITS_DIR / "train_split.csv",
    "Validation": SPLITS_DIR / "val_split.csv",
    "Test": SPLITS_DIR / "test_split.csv",
}

OUTPUT_DIR = OUTPUT_PLOT / "dataset_distribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_ORDER = ["Train", "Validation", "Test"]
CLASS_ORDER = ["Benign", "Malignant"]

DENSITY_LABELS = {
    1: "1 - Almost entirely fatty",
    2: "2 - Scattered fibroglandular tissue",
    3: "3 - Heterogeneously dense",
    4: "4 - Extremely dense",
}


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts column names to lowercase snake_case.
    Example:
        'breast density' -> 'breast_density'
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    column_description: str,
) -> str:
    """
    Finds the first available column among a list of candidates.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"Could not find the {column_description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {df.columns.tolist()}"
    )


def normalize_label(value) -> str:
    """
    Converts numerical or textual labels to Benign/Malignant.
    """
    if pd.isna(value):
        return "Missing"

    value_as_string = str(value).strip().upper()

    benign_values = {
        "0",
        "0.0",
        "BENIGN",
        "BENIGN_WITHOUT_CALLBACK",
        "FALSE",
    }

    malignant_values = {
        "1",
        "1.0",
        "MALIGNANT",
        "TRUE",
    }

    if value_as_string in benign_values:
        return "Benign"

    if value_as_string in malignant_values:
        return "Malignant"

    raise ValueError(
        f"Unrecognised class label: {value!r}"
    )


def normalize_density(value) -> str:
    """
    Converts breast-density values 1-4 to readable labels.
    Unknown values are preserved.
    """
    if pd.isna(value):
        return "Missing"

    value_as_string = str(value).strip()

    try:
        density_number = int(float(value_as_string))

        if density_number in DENSITY_LABELS:
            return DENSITY_LABELS[density_number]

    except ValueError:
        pass

    return value_as_string


def annotate_stacked_bars(
    ax: plt.Axes,
    percentages: pd.DataFrame,
    minimum_visible_percentage: float = 4.0,
) -> None:
    """
    Adds percentage labels inside stacked bars.
    """
    cumulative_values = np.zeros(len(percentages))

    for class_name in CLASS_ORDER:
        values = percentages[class_name].to_numpy()

        for index, value in enumerate(values):
            if value >= minimum_visible_percentage:
                ax.text(
                    index,
                    cumulative_values[index] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        cumulative_values += values


split_dataframes = []

for split_name, file_path in SPLIT_FILES.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing split file: {file_path}"
        )

    split_df = pd.read_csv(file_path)
    split_df = standardize_column_names(split_df)

    label_column = find_column(
        split_df,
        candidates=[
            "label",
            "target",
            "class",
            "pathology_label",
            "binary_label",
        ],
        column_description="label",
    )

    density_column = find_column(
        split_df,
        candidates=[
            "breast_density",
            "density",
            "breast_density_value",
        ],
        column_description="breast density",
    )

    selected_df = split_df[
        [label_column, density_column]
    ].copy()

    selected_df["split"] = split_name

    selected_df["class"] = (
        selected_df[label_column]
        .apply(normalize_label)
    )

    selected_df["breast_density"] = (
        selected_df[density_column]
        .apply(normalize_density)
    )

    split_dataframes.append(
        selected_df[
            ["split", "class", "breast_density"]
        ]
    )


distribution_df = pd.concat(
    split_dataframes,
    ignore_index=True,
)

distribution_df["split"] = pd.Categorical(
    distribution_df["split"],
    categories=SPLIT_ORDER,
    ordered=True,
)


missing_labels = distribution_df["class"].eq("Missing").sum()
missing_densities = distribution_df["breast_density"].eq("Missing").sum()

print(f"Total observations: {len(distribution_df)}")
print(f"Missing labels: {missing_labels}")
print(f"Missing breast-density values: {missing_densities}")

if missing_labels > 0:
    raise ValueError(
        "Some observations have no valid class label."
    )


overall_counts = pd.crosstab(
    distribution_df["split"],
    distribution_df["class"],
).reindex(
    index=SPLIT_ORDER,
    columns=CLASS_ORDER,
    fill_value=0,
)

overall_percentages = (
    overall_counts
    .div(overall_counts.sum(axis=1), axis=0)
    .mul(100)
)


print("\nOverall counts by split")
print(overall_counts)

print("\nOverall percentages by split")
print(overall_percentages.round(2))


density_counts = (
    distribution_df
    .groupby(
        ["split", "breast_density", "class"],
        observed=True,
    )
    .size()
    .unstack(
        fill_value=0,
    )
    .reindex(
        columns=CLASS_ORDER,
        fill_value=0,
    )
)

density_percentages = (
    density_counts
    .div(density_counts.sum(axis=1), axis=0)
    .mul(100)
)


print("\nCounts by split, breast density and class")
print(density_counts)

print("\nPercentages by split and breast density")
print(density_percentages.round(2))


summary_table = (
    density_counts
    .reset_index()
)

summary_table["Total"] = (
    summary_table[CLASS_ORDER]
    .sum(axis=1)
)

for class_name in CLASS_ORDER:
    summary_table[f"{class_name}_percentage"] = (
        summary_table[class_name]
        .div(summary_table["Total"])
        .mul(100)
        .round(2)
    )


print("\nDetailed summary")
print(summary_table.to_string(index=False))


# Save summary tables
overall_counts.to_csv(
    OUTPUT_DIR / "overall_class_counts_local.csv"
)

overall_percentages.round(2).to_csv(
    OUTPUT_DIR / "overall_class_percentages_local.csv"
)

summary_table.to_csv(
    OUTPUT_DIR / "class_distribution_by_density_local.csv",
    index=False,
)


fig, ax = plt.subplots(figsize=(8, 6))

bottom = np.zeros(len(overall_percentages))

for class_name in CLASS_ORDER:
    values = overall_percentages[class_name].to_numpy()

    ax.bar(
        overall_percentages.index,
        values,
        bottom=bottom,
        label=class_name,
    )

    bottom += values

annotate_stacked_bars(
    ax,
    overall_percentages,
)

ax.set_title(
    "Benign and malignant class distribution by local dataset split"
)
ax.set_xlabel("Dataset split")
ax.set_ylabel("Proportion of observations (%)")
ax.set_ylim(0, 100)
ax.legend(title="Class")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "class_distribution_by_split_local.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


available_densities = [
    density
    for density in DENSITY_LABELS.values()
    if density in distribution_df["breast_density"].unique()
]

if "Missing" in distribution_df["breast_density"].unique():
    available_densities.append("Missing")

other_densities = sorted(
    set(distribution_df["breast_density"].unique())
    - set(available_densities)
)

available_densities.extend(other_densities)

complete_index = pd.MultiIndex.from_product(
    [SPLIT_ORDER, available_densities],
    names=["split", "breast_density"],
)

density_counts_for_plot = density_counts.reindex(
    complete_index,
    fill_value=0,
)

density_percentages_for_plot = (
    density_counts_for_plot
    .div(
        density_counts_for_plot.sum(axis=1).replace(0, np.nan),
        axis=0,
    )
    .mul(100)
    .fillna(0)
)


plot_labels = [
    f"{split_name}\n{density_name.split(' - ')[0]}"
    for split_name, density_name
    in density_percentages_for_plot.index
]

fig, ax = plt.subplots(figsize=(14, 7))

x_positions = np.arange(
    len(density_percentages_for_plot)
)

bottom = np.zeros(
    len(density_percentages_for_plot)
)

for class_name in CLASS_ORDER:
    values = (
        density_percentages_for_plot[class_name]
        .to_numpy()
    )

    bars = ax.bar(
        x_positions,
        values,
        bottom=bottom,
        label=class_name,
    )

    for bar, value, base in zip(
        bars,
        values,
        bottom,
    ):
        if value >= 4:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    bottom += values

ax.set_title(
    "Benign and malignant proportions by breast density"
)
ax.set_xlabel(
    "Dataset split and breast-density category"
)
ax.set_ylabel("Proportion of observations (%)")
ax.set_ylim(0, 100)

ax.set_xticks(x_positions)
ax.set_xticklabels(
    plot_labels,
    rotation=45,
    ha="right",
)

ax.legend(title="Class")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "class_proportion_by_density_local.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

density_totals = (
    distribution_df
    .groupby(
        ["breast_density", "split"],
        observed=True,
    )
    .size()
    .unstack(
        fill_value=0,
    )
    .reindex(
        index=available_densities,
        columns=SPLIT_ORDER,
        fill_value=0,
    )
)

fig, ax = plt.subplots(figsize=(11, 6))

x_positions = np.arange(
    len(density_totals.index)
)

bar_width = 0.25

for split_index, split_name in enumerate(SPLIT_ORDER):
    offsets = (
        x_positions
        + (split_index - 1) * bar_width
    )

    values = density_totals[split_name].to_numpy()

    bars = ax.bar(
        offsets,
        values,
        width=bar_width,
        label=split_name,
    )

    ax.bar_label(
        bars,
        padding=3,
        fontsize=8,
    )

ax.set_title(
    "Number of observations by breast density and dataset split"
)
ax.set_xlabel("Breast-density category")
ax.set_ylabel("Number of observations")

ax.set_xticks(x_positions)
ax.set_xticklabels(
    [
        density.replace(" - ", "\n")
        for density in density_totals.index
    ],
)

ax.legend(title="Dataset split")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "sample_count_by_density_local.png",
    dpi=300,
    bbox_inches="tight",
)


fig, ax = plt.subplots(figsize=(8, 6))

overall_percentages[
    ["Benign", "Malignant"]
].plot(
    kind="bar",
    ax=ax,
)

ax.set_title(
    "Benign and malignant class distribution by local dataset split"
)
ax.set_xlabel("Dataset split")
ax.set_ylabel("Proportion of observations (%)")
ax.set_ylim(0, 100)
ax.legend(title="Class")
ax.grid(axis="y", alpha=0.3)

# Display percentage above each bar
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.1f%%",
        padding=3,
    )

plt.xticks(rotation=0)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "class_distribution_by_split_local.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()