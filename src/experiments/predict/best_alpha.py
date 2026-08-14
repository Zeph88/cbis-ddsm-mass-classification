import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from src.training.dataset_preparation import train_val_test_sets

from src.config import OUTPUT_MODEL, OUTPUT_NPY
from src.functions import ensure_directory

def collect_branch_predictions(
    dataset,
    local_model,
    global_model,
):
    """
    Collect the labels and the predictions produced by both branches.

    Both predictions are computed during the same dataset iteration to
    preserve the exact correspondence between samples.
    """

    y_true = []
    local_probabilities = []
    global_probabilities = []

    for images, labels in dataset:
        local_images, global_images = images

        local_probs = local_model(
            local_images,
            training=False,
        ).numpy().ravel()

        global_probs = global_model(
            global_images,
            training=False,
        ).numpy().ravel()

        y_true.extend(labels.numpy().ravel())
        local_probabilities.extend(local_probs)
        global_probabilities.extend(global_probs)

    return (
        np.asarray(y_true),
        np.asarray(local_probabilities),
        np.asarray(global_probabilities),
    )

ensure_directory(OUTPUT_MODEL)

param_inputs = {
    'zoom':{
        'width':256,
        'height':256
    },
    'full':{
        'width':512,
        'height':768
    }
}

local_model = tf.keras.models.load_model(
    OUTPUT_MODEL / "model_local_branch.keras"
)

global_model = tf.keras.models.load_model(
    OUTPUT_MODEL / "model_global_branch.keras"
)


local_df = pd.read_csv(OUTPUT_NPY / f"dataset_index_zoom_{param_inputs['zoom']['height']}x{param_inputs['zoom']['width']}.csv")
global_df = pd.read_csv(OUTPUT_NPY / f"dataset_index_full_{param_inputs['full']['height']}x{param_inputs['full']['width']}.csv")

common_id = set(local_df['lesion_key']) & set(global_df['lesion_key'])
local_df = local_df[local_df['lesion_key'].isin(common_id)]
global_df = global_df[global_df['lesion_key'].isin(common_id)]
local_df['local_path'] = local_df['preprocessed_image_path']
global_df['global_path'] = global_df['preprocessed_image_path']
df = pd.merge(local_df, global_df[['global_path', 'lesion_key']], on='lesion_key')
print(df)

train_df, val_df, test_df = train_val_test_sets(df, path_image='local_path', added_path_image='global_path', image_height=param_inputs['zoom']['height'], 
    image_width=param_inputs['zoom']['width'], added_image_height=param_inputs['full']['height'], added_image_width=param_inputs['full']['width'])


val_true, val_local_prob, val_global_prob = (
    collect_branch_predictions(
        val_df,
        local_model,
        global_model,
    )
)


# Test weights from 0% to 100% global contribution
alphas = np.linspace(0.0, 1.0, 101)

validation_aucs = []

for alpha in alphas:
    fused_probabilities = (
        alpha * val_global_prob
        + (1.0 - alpha) * val_local_prob
    )

    auc = roc_auc_score(
        val_true,
        fused_probabilities,
    )

    validation_aucs.append(auc)

validation_aucs = np.asarray(validation_aucs)

best_index = int(
    np.argmax(validation_aucs)
)

best_alpha = float(
    alphas[best_index]
)

best_validation_auc = float(
    validation_aucs[best_index]
)

print(
    f"Best global weight: {best_alpha:.2f}"
)

print(
    f"Best local weight: {1.0 - best_alpha:.2f}"
)

print(
    f"Best validation AUC: "
    f"{best_validation_auc:.4f}"
)

import matplotlib.pyplot as plt


plt.figure(figsize=(9, 6))

plt.plot(
    alphas,
    validation_aucs,
    linewidth=2,
    label="Validation AUC",
)

# Highlight the equal-weight baseline
equal_weight_index = int(
    np.argmin(np.abs(alphas - 0.5))
)

equal_weight_auc = validation_aucs[
    equal_weight_index
]

plt.scatter(
    0.5,
    equal_weight_auc,
    s=90,
    label=(
        f"Equal weighting: "
        f"AUC = {equal_weight_auc:.4f}"
    ),
    zorder=3,
)

# Highlight the best validation weight
plt.scatter(
    best_alpha,
    best_validation_auc,
    s=110,
    label=(
        f"Best weighting: "
        f"global = {best_alpha:.2f}, "
        f"local = {1.0 - best_alpha:.2f}, "
        f"AUC = {best_validation_auc:.4f}"
    ),
    zorder=4,
)

plt.axvline(
    best_alpha,
    linestyle="--",
    alpha=0.6,
)

plt.xlabel(
    "Global-model weight α"
)

plt.ylabel(
    "Validation AUC"
)

plt.title(
    "Validation AUC by local–global probability weighting"
)

plt.xlim(0, 1)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT
    / "probability_fusion_auc_by_alpha.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()

test_true, test_local_prob, test_global_prob = (
    collect_branch_predictions(
        test_df,
        local_model,
        global_model,
    )
)

test_fused_prob = (
    best_alpha * test_global_prob
    + (1.0 - best_alpha) * test_local_prob
)

test_auc = roc_auc_score(
    test_true,
    test_fused_prob,
)

local_test_auc = roc_auc_score(
    test_true,
    test_local_prob,
)

global_test_auc = roc_auc_score(
    test_true,
    test_global_prob,
)

equal_test_prob = (
    0.5 * test_global_prob
    + 0.5 * test_local_prob
)

equal_test_auc = roc_auc_score(
    test_true,
    equal_test_prob,
)

print(f"Local test AUC: {local_test_auc:.4f}")
print(f"Global test AUC: {global_test_auc:.4f}")
print(f"Equal-weight test AUC: {equal_test_auc:.4f}")

print(
    f"Weighted-fusion test AUC: {test_auc:.4f} "
    f"(global={best_alpha:.2f}, "
    f"local={1.0 - best_alpha:.2f})"
)