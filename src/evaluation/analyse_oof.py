from sklearn.calibration import (
    calibration_curve,
)
import matplotlib.pyplot as plt

def patient_cluster_bootstrap(
    df,
    n_bootstrap=10_000,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    groups = {
        patient_id: group.copy()
        for patient_id, group
        in df.groupby(
            "patient_id",
            sort=False,
        )
    }

    patient_ids = np.asarray(
        list(groups)
    )

    rows = []

    for iteration in range(
        n_bootstrap
    ):
        sampled_ids = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )

        sampled = pd.concat(
            [
                groups[
                    patient_id
                ]
                for patient_id
                in sampled_ids
            ],
            ignore_index=True,
        )

        y = (
            sampled["label"]
            .astype(int)
            .to_numpy()
        )

        if np.unique(y).size < 2:
            continue

        sym = sampled[
            "symmetric_probability"
        ].to_numpy()

        res = sampled[
            "residual_probability"
        ].to_numpy()

        sym_auc = roc_auc_score(
            y,
            sym,
        )

        res_auc = roc_auc_score(
            y,
            res,
        )

        sym_bce = log_loss(
            y,
            sym,
            labels=[0, 1],
        )

        res_bce = log_loss(
            y,
            res,
            labels=[0, 1],
        )

        rows.append(
            {
                "delta_auc":
                    res_auc
                    - sym_auc,

                "delta_bce":
                    res_bce
                    - sym_bce,
            }
        )

    return pd.DataFrame(
        rows
    )

def calibration_values(
    y_true,
    probability,
):
    observed, predicted = (
        calibration_curve(
            y_true,
            probability,
            n_bins=5,
            strategy="quantile",
        )
    )

    brier = brier_score_loss(
        y_true,
        probability,
    )

    return (
        observed,
        predicted,
        brier,
    )

bootstrap_df = (
    patient_cluster_bootstrap(
        oof_df,
        n_bootstrap=10_000,
        seed=42,
    )
)

auc_low = (
    bootstrap_df[
        "delta_auc"
    ]
    .quantile(0.025)
)

auc_high = (
    bootstrap_df[
        "delta_auc"
    ]
    .quantile(0.975)
)

print(
    "Delta AUC 95% CI:",
    auc_low,
    auc_high,
)

y = (
    oof_df["label"]
    .astype(int)
    .to_numpy()
)

sym = (
    oof_df[
        "symmetric_probability"
    ]
    .to_numpy()
)

res = (
    oof_df[
        "residual_probability"
    ]
    .to_numpy()
)

(
    sym_observed,
    sym_predicted,
    sym_brier,
) = calibration_values(
    y,
    sym,
)

(
    res_observed,
    res_predicted,
    res_brier,
) = calibration_values(
    y,
    res,
)

fig, ax = plt.subplots(
    figsize=(6, 6)
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

ax.plot(
    sym_predicted,
    sym_observed,
    marker="o",
    label=(
        f"Symmetric "
        f"(Brier={sym_brier:.3f})"
    ),
)

ax.plot(
    res_predicted,
    res_observed,
    marker="o",
    label=(
        f"Residual "
        f"(Brier={res_brier:.3f})"
    ),
)

ax.set_xlabel(
    "Mean predicted malignancy probability"
)

ax.set_ylabel(
    "Observed malignant proportion"
)

ax.set_title(
    "OOF reliability diagram"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    RESULTS_DIR
    / "oof_reliability_diagram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)