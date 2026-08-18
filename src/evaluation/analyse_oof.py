import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss,roc_auc_score


from src.config import OUTPUT_MODEL, SEED, N_OUTER_FOLDS, ARCHITECTURES, N_BOOTSTRAP, N_CALIBRATION_BINS
from src.functions import ensure_directory
from src.evaluation.evaluation_utils import calculate_metrics

# Configuration
METRIC_NAMES = ["auc", "bce", "ap", "brier"]

# First model minus second model.
PAIRWISE_COMPARISONS = [
    ("residual", "symmetric"),
    ("symmetric", "local"),
    ("residual", "local"),
    ("symmetric", "global"),
    ("residual", "global"),
    ("local", "global"),
]

CV_ROOT = OUTPUT_MODEL / f"fusion_cv_{N_OUTER_FOLDS}fold_seed_{SEED}"
FOLDS_DIR = CV_ROOT / "folds"
RESULTS_DIR = CV_ROOT / "results"

ensure_directory(RESULTS_DIR)


# Load OOF predictions
def load_fold_predictions(fold):
    fold_dir = (FOLDS_DIR / f"fold_{fold}")
    predictions_path = (fold_dir / "oof_predictions.csv")

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing OOF predictions: "
            f"{predictions_path}"
        )

    fold_df = pd.read_csv(predictions_path)

    # Try to merge separately generated branch predictions.
    missing_branch_columns = {
        "local_probability",
        "global_probability",
    } - set(fold_df.columns)

    if missing_branch_columns:
        branch_path = (fold_dir / "branch_oof_predictions.csv")

        if not branch_path.exists():
            raise FileNotFoundError(
                "\nLocal/global OOF probabilities are missing.\n"
                f"Fold: {fold}\n"
                f"Missing columns: "
                f"{sorted(missing_branch_columns)}\n\n"
                "Generate predictions from the saved local and "
                "global checkpoints on the SAME outer-evaluation "
                "observations and save them as:\n"
                f"{branch_path}\n\n"
                "Required columns:\n"
                "sample_id, patient_id, fold, label, "
                "local_probability, global_probability"
            )

        branch_df = pd.read_csv(branch_path)

        # merge_keys = ["sample_id", "patient_id", "fold", "label"]
        merge_keys = ["sample_id", "fold"]

        fold_df = fold_df.merge(
            branch_df[
                merge_keys
                + [
                    "local_probability",
                    "global_probability",
                ]
            ],
            on=merge_keys, how="left", validate="one_to_one"
        )

    return fold_df


def load_cv_outputs():
    frames = []

    for fold in range(N_OUTER_FOLDS):
        frames.append(load_fold_predictions(fold))

    return pd.concat(frames, ignore_index=True)


# Integrity checks
def validate_oof(oof_df):
    required_columns = {
        "sample_id", 
        "patient_id",
        "fold",
        "label",
        "local_probability",
        "global_probability",
        "symmetric_probability",
        "residual_probability",
    }

    missing = (required_columns - set(oof_df.columns))

    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")

    expected_folds = set(range(N_OUTER_FOLDS))
    observed_folds = set(oof_df["fold"].astype(int).unique())

    if observed_folds != expected_folds:
        raise ValueError(
            f"Unexpected outer folds. Expected {expected_folds}, received {observed_folds}."
        )

    # Each lesion/sample must appear exactly once OOF.
    if oof_df["sample_id"].duplicated().any():
        duplicated = oof_df.loc[oof_df["sample_id"].duplicated(keep=False), ["sample_id", "fold"]]

        raise ValueError(
            f"Some samples occur in more than one OOF row: {duplicated.head()}"
        )

    # A patient must belong to one outer evaluation fold only.
    patient_fold_counts = (oof_df.groupby("patient_id")["fold"].nunique())

    leaking_patients = (patient_fold_counts[patient_fold_counts > 1])

    if not leaking_patients.empty:
        raise ValueError(
            f"Some patients occur in several outer evaluation folds: {leaking_patients.head()}"
        )

    labels = set(oof_df["label"].astype(int).unique())

    if labels != {0, 1}:
        raise ValueError("OOF predictions must contain both classes.")

    for architecture in (ARCHITECTURES):
        column = (f"{architecture}_probability")

        if oof_df[column].isna().any():
            raise ValueError(f"Missing probabilities for {architecture}.")

        probability = (oof_df[column].to_numpy())

        if (probability < 0).any() or (probability > 1).any():
            raise ValueError(f"Probabilities outside [0, 1] for {architecture}.")

    print("OOF integrity checks passed.")
    print(f"OOF observations: {len(oof_df)}")
    print("OOF patients:", oof_df["patient_id"].nunique())


# Fold-level metrics
def calculate_fold_metrics(oof_df):
    
    rows = []
    for fold in sorted(oof_df["fold"].unique()):
        fold_df = (oof_df[oof_df["fold"] == fold])
        y_true = (fold_df["label"].astype(int).to_numpy())

        for architecture in ARCHITECTURES:
            probability = (fold_df[f"{architecture}_probability"].to_numpy())
            scores = calculate_metrics(y_true, probability)

            rows.append(
                {
                    "fold": int(fold),
                    "seed": SEED,
                    "architecture": architecture,
                    "n_outer_evaluation": len(fold_df),
                    "n_outer_evaluation_patients": fold_df["patient_id"].nunique(), 
                    **scores
                }
            )

    return pd.DataFrame(rows)


def summarize_fold_metrics(metrics_df):
    
    summary = (metrics_df.groupby("architecture")[METRIC_NAMES].agg(["mean", "std", "min", "max"]))
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]

    return (summary.reset_index())


# Paired fold differences
def calculate_fold_differences(metrics_df):

    pivot = metrics_df.pivot(index="fold", columns="architecture", values=METRIC_NAMES)

    rows = []
    for fold in pivot.index:
        for first, second in (PAIRWISE_COMPARISONS):
            row = {
                "fold": int(fold),
                "comparison":
                    f"{first}_minus_{second}",
            }

            for metric in (METRIC_NAMES):
                row[f"delta_{metric}"] = (pivot.loc[fold, (metric, first)]
                    - pivot.loc[fold, (metric, second)])

            rows.append(row)

    return pd.DataFrame(rows)


# Pooled OOF metrics
def calculate_pooled_metrics(oof_df):

    y_true = (oof_df["label"].astype(int).to_numpy())
    prevalence = float(y_true.mean())

    baseline_probability = np.full(len(y_true), prevalence, dtype=np.float64)
    baseline_scores = (calculate_metrics(y_true, baseline_probability))

    rows = []
    for architecture in (ARCHITECTURES):

        probability = (oof_df[f"{architecture}_probability"].to_numpy())
        scores = calculate_metrics(y_true, probability)
        brier_skill_score = (1.0 - scores["brier"] / baseline_scores["brier"])

        rows.append(
            {
                "architecture": architecture,
                "n_observations": len(oof_df),
                "n_patients": oof_df["patient_id"].nunique(),
                "prevalence": prevalence,
                **scores,
                "brier_skill_score": brier_skill_score,
            }
        )

    rows.append(
        {
            "architecture": "prevalence_baseline",
            "n_observations": len(oof_df),
            "n_patients": oof_df["patient_id"].nunique(),
            "prevalence": prevalence,
            **baseline_scores,
            "brier_skill_score": 0.0,
        }
    )

    return pd.DataFrame(rows)


# Quantities used by bootstrap
def calculate_all_quantities(df):

    y_true = (df["label"].astype(int).to_numpy())

    architecture_scores = {}
    quantities = {}

    for architecture in (ARCHITECTURES):

        probability = (df[f"{architecture}_probability"].to_numpy())
        scores = calculate_metrics(y_true, probability)
        architecture_scores[architecture] = scores

        for metric in (METRIC_NAMES):
            quantities[f"{architecture}_{metric}"] = scores[metric]

    # Paired model differences.
    for first, second in (PAIRWISE_COMPARISONS):
        for metric in (METRIC_NAMES):
            quantities[f"delta_{first}_minus_{second}_{metric}"] = (architecture_scores[first][metric] - architecture_scores[second][metric])

    return quantities


# Patient-cluster bootstrap
def patient_cluster_bootstrap(oof_df, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    
    oof_df = (oof_df.reset_index(drop=True))
    rng = np.random.default_rng(seed)

    patient_groups = [group.index.to_numpy() for _, group in oof_df.groupby("patient_id", sort=False)]
    n_patients = len(patient_groups)

    rows = []
    for iteration in range(n_bootstrap):

        sampled_patients = (rng.integers(low=0, high=n_patients, size=n_patients))
        sampled_rows = (np.concatenate([patient_groups[patient_index] for patient_index in sampled_patients]))
        bootstrap_sample = (oof_df.iloc[sampled_rows])

        # ROC-AUC requires both classes.
        if (bootstrap_sample["label"].nunique() < 2):
            continue

        quantities = (calculate_all_quantities(bootstrap_sample))

        rows.append(
            {
                "iteration": iteration,
                **quantities,
            }
        )

    bootstrap_df = pd.DataFrame(rows)

    if bootstrap_df.empty:
        raise RuntimeError("No valid bootstrap replicates generated.")

    return bootstrap_df


def summarize_bootstrap(oof_df, bootstrap_df):

    point_estimates = (calculate_all_quantities(oof_df))

    rows = []
    for quantity, estimate in (point_estimates.items()):
        rows.append(
            {
                "quantity": quantity,
                "estimate": estimate,
                "ci_2.5": bootstrap_df[quantity].quantile(0.025),
                "ci_97.5": bootstrap_df[quantity].quantile(0.975)
            }
        )

    return pd.DataFrame(rows)


# Calibration
def build_calibration_table(y_true, probability, architecture, n_bins=N_CALIBRATION_BINS):

    calibration_df = pd.DataFrame(
        {
            "label": y_true,
            "probability": probability
        }
    )

    calibration_df["quantile_bin"] = pd.qcut(
        calibration_df["probability"], q=n_bins, duplicates="drop"
    )

    table = (
        calibration_df.groupby("quantile_bin", observed=True).agg(
            count=("label", "size"),
            mean_probability=("probability", "mean"),
            observed_fraction=("label", "mean"),
        ).reset_index(drop=True)
    )

    table.insert(0, "bin", np.arange(1, len(table) + 1))
    table.insert(0, "architecture", architecture)
    table["absolute_gap"] = np.abs(table["observed_fraction"] - table["mean_probability"])

    return table

def calculate_ece(calibration_table):
    return float(np.average(calibration_table["absolute_gap"], weights=calibration_table["count"]))

def create_reliability_diagram(calibration_tables):

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for architecture in (ARCHITECTURES):
        table = (calibration_tables[architecture])
        ece = calculate_ece(table)

        ax.plot(
            table["mean_probability"],
            table["observed_fraction"],
            marker="o",
            label=(f"{architecture.capitalize()} (ECE={ece:.3f})"),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Mean predicted malignancy probability")
    ax.set_ylabel("Observed malignant proportion")
    ax.set_title("OOF reliability diagram")
    ax.legend()
    fig.tight_layout()

    fig.savefig(RESULTS_DIR / "oof_reliability_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():

    print("Loading OOF predictions...")

    oof_df = load_cv_outputs()

    validate_oof(oof_df)

    # Fold metrics calciulations
    fold_metrics = (calculate_fold_metrics(oof_df))
    fold_summary = (summarize_fold_metrics(fold_metrics))
    fold_differences = (calculate_fold_differences(fold_metrics))

    oof_df.to_csv(RESULTS_DIR / "all_oof_predictions.csv", index=False)
    fold_metrics.to_csv(RESULTS_DIR / "all_fold_metrics.csv", index=False)
    fold_summary.to_csv(RESULTS_DIR / "fold_metric_summary.csv", index=False)
    fold_differences.to_csv(RESULTS_DIR / "fold_pairwise_differences.csv", index=False)

    print("\nFold-level summary")
    print(fold_summary.to_string(index=False))
    print("\nPaired fold differences")
    print(fold_differences.to_string(index=False))

    # Pooled OOF metrics
    pooled_metrics = (calculate_pooled_metrics(oof_df))
    pooled_metrics.to_csv(RESULTS_DIR / "pooled_oof_metrics.csv", index=False)
    print("\nPooled OOF metrics")
    print(pooled_metrics.to_string(index=False))

    # Patient-cluster bootstrap
    print(f"\nRunning patient-cluster bootstrap ({N_BOOTSTRAP} replicates)...")

    bootstrap_df = (patient_cluster_bootstrap(oof_df, n_bootstrap=N_BOOTSTRAP, seed=SEED))
    bootstrap_summary = (summarize_bootstrap(oof_df, bootstrap_df))

    bootstrap_df.to_csv(RESULTS_DIR / "patient_cluster_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(RESULTS_DIR / "bootstrap_95ci_summary.csv", index=False)
    print("\nBootstrap 95% confidence intervals")
    print(bootstrap_summary.to_string(index=False))

    # Calibration*
    y_true = (oof_df["label"].astype(int).to_numpy())

    calibration_tables = {}
    calibration_frames = []

    for architecture in (ARCHITECTURES):
        probability = (oof_df[f"{architecture}_probability"].to_numpy())

        table = build_calibration_table(y_true, probability, architecture)
        calibration_tables[architecture] = table
        calibration_frames.append(table)

        print(f"{architecture.capitalize()} ECE: {calculate_ece(table):.4f}")

    pd.concat(calibration_frames, ignore_index=True).to_csv(RESULTS_DIR / "oof_calibration_bins.csv", index=False)

    create_reliability_diagram(calibration_tables)

    print("\nAnalysis complete.")
    print(f"Results written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()