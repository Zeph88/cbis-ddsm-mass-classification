# CBIS-DDSM Mass Classification

Deep-learning project for **benign vs malignant classification of known mammographic masses** using the CBIS-DDSM dataset.

The project investigates whether a lesion-centred representation benefits from complementary information extracted from the full mammogram. Four architectures are evaluated:

- a **local ResNet50 branch** trained on ROI-centred lesion crops;
- a **global ResNet50 branch** trained on full mammograms;
- a **symmetric local-global fusion model** that concatenates local and global embeddings;
- a **residual local-global fusion model** that preserves the local decision as a baseline and learns a contextual correction from local and global features.

Grad-CAM is used for qualitative interpretation of the local, global and residual-fusion models.

> **Academic project only.** This repository is intended for research and coursework. It is not a validated medical device and must not be used for clinical decision-making.

---

## Research question

The main question is:

> **Does combining local lesion information with global mammographic context improve benign–malignant mass classification compared with either representation alone?**

A secondary question examines **how** the two representations should be combined: symmetrically, or through a residual formulation in which the local lesion prediction remains the primary decision and global context learns only a correction.

---

## Dataset and task

The project uses the **mass subset of CBIS-DDSM**.

The task is **computer-aided diagnosis (CADx)** rather than lesion detection. The abnormality is assumed to be known from the expert-provided ROI mask, and the target is binary:

- `0`: benign / benign without callback;
- `1`: malignant.

Raw DICOM files are **not included in the repository** and must be obtained separately.

The original metadata used by the project are version-controlled under:

```text
data/metadata/
├── mass_case_description_train_set.csv
└── mass_case_description_test_set.csv
```

The resolved lesion-level dataset index is stored under:

```text
data/processed/mass_dataset_index.csv
```

---

## Experimental design

Patient identity is treated as the grouping variable throughout the evaluation pipeline to reduce the risk of patient leakage.

The project contains two complementary evaluation settings:

1. **Fixed train / validation / test partitions** for model development, saved-model evaluation and operating-threshold selection.
2. **Five-fold patient-grouped cross-validation** for the final comparison of local, global, symmetric-fusion and residual-fusion architectures.

The same outer patient partitions are used for all architectures in the cross-validation analysis, allowing paired comparison on matched out-of-fold observations.

For uncertainty estimation, paired model differences are evaluated using a **patient-cluster bootstrap with 10,000 replicates** and percentile-based 95% confidence intervals.

---

## Reported results

The values below reproduce the results reported in the accompanying `draft_report.pdf`. Small numerical differences may occur when neural-network training is repeated, even with fixed random seeds.

### Five-fold patient-grouped cross-validation

Mean ± standard deviation across five outer folds:

| Architecture | ROC-AUC | BCE | AP | Brier |
|---|---:|---:|---:|---:|
| Global | 0.651 ± 0.073 | 0.679 ± 0.043 | 0.643 ± 0.084 | 0.240 ± 0.019 |
| Local | 0.692 ± 0.106 | 0.632 ± 0.049 | 0.694 ± 0.121 | 0.222 ± 0.023 |
| Symmetric fusion | 0.723 ± 0.092 | 0.623 ± 0.080 | 0.728 ± 0.100 | 0.216 ± 0.034 |
| **Residual fusion** | **0.734 ± 0.076** | **0.602 ± 0.056** | **0.742 ± 0.093** | **0.208 ± 0.025** |

Residual fusion achieved the strongest mean performance across the four reported metrics, although performance varied substantially between patient partitions.

### Pooled out-of-fold results

The pooled OOF ROC-AUC values were:

| Architecture | Pooled ROC-AUC |
|---|---:|
| Global | 0.646 |
| Local | 0.697 |
| Symmetric fusion | 0.724 |
| **Residual fusion** | **0.733** |

For residual fusion, the pooled OOF metrics were:

```text
ROC-AUC: 0.733
AP:      0.745
BCE:     0.602
Brier:   0.208
```

### Paired patient-level bootstrap

Residual fusion compared with the local branch:

| Metric | Residual − Local | 95% bootstrap CI |
|---|---:|---:|
| ROC-AUC | +0.036 | [+0.012, +0.060] |
| BCE | -0.030 | [-0.048, -0.011] |
| AP | +0.038 | [+0.016, +0.059] |
| Brier | -0.013 | [-0.022, -0.005] |

All four intervals excluded zero in the favourable direction.

The residual-vs-symmetric comparison was more nuanced. The ROC-AUC difference was small and its confidence interval included zero:

```text
Residual − Symmetric ROC-AUC:
+0.009  [95% CI: -0.008, +0.026]
```

However, residual fusion showed clearer improvements in BCE and average precision:

```text
BCE:   -0.021  [95% CI: -0.039, -0.003]
AP:    +0.032  [95% CI: +0.005, +0.059]
Brier: -0.007  [95% CI: -0.014, +0.0001]
```

### Calibration

Expected calibration error (ECE) calculated from pooled OOF predictions:

| Architecture | ECE |
|---|---:|
| Global | 0.0797 |
| Local | 0.0589 |
| Symmetric fusion | 0.0540 |
| **Residual fusion** | **0.0240** |

ECE is interpreted descriptively because it depends on the calibration-bin definition.

### Operating threshold reported in the draft

The retained residual-fusion operating threshold was selected from **validation predictions only**, with a predefined target of at least **85% sensitivity**.

The threshold reported in the draft was:

```text
Selected threshold: 0.265
```

Applied unchanged to the paired test set, it produced:

| Metric | Test result |
|---|---:|
| Sensitivity / recall | 0.884 |
| Specificity | 0.537 |
| Precision | 0.563 |
| F1-score | 0.688 |
| Balanced accuracy | 0.711 |
| Accuracy | 0.677 |
| True positives | 130 |
| True negatives | 117 |
| False positives | 101 |
| False negatives | 17 |

The threshold is an **operating point**, not an intrinsic model parameter. Re-training the model may shift the probability distribution and therefore the validation-selected threshold.

---

## Model architecture

### Local branch

The local branch receives a lesion-centred mammographic crop:

```text
Input: 384 × 384 × 1
```

The expert ROI mask is used to identify the lesion and centre the crop. The final model uses an ImageNet-pretrained ResNet50 backbone with frozen convolutional weights and a lightweight task-specific classification head.

The local branch is the strongest individual branch in the final comparison.

### Global branch

The global branch receives the full preprocessed mammogram:

```text
Input: 768 × 512 × 1
```

The preprocessing normalises breast orientation, removes isolated annotations and background artefacts, and retains the main breast region.

The global branch is weaker as a standalone classifier than the local branch but provides complementary contextual information to the fusion models.

### Symmetric fusion

The symmetric architecture retrieves learned embeddings from the frozen local and global branches, normalises them independently, concatenates them, and trains a lightweight fusion head.

Conceptually:

```text
local embedding ─┐
                 ├─ concatenate → fusion head → probability
global embedding ┘
```

### Residual fusion

The residual architecture treats the local prediction as the baseline and learns a contextual logit correction:

```text
final logit = frozen local logit + contextual correction
```

The correction is learned from the combined local and global embeddings.

The final correction layer is zero-initialised so that, before fusion training:

```text
residual prediction = local prediction
```

This makes the global representation complementary rather than forcing it to contribute symmetrically to the final decision.

---

## Preprocessing

Preprocessing is performed once and saved as NumPy arrays to avoid repeated DICOM decoding during training.

### Local preprocessing

```bash
python -m src.preprocessing.dataset_preprocessing --mode local
```

This generates:

- lesion-centred `384 × 384` local images;
- lesion-specific ROI masks transformed into the same `768 × 512` spatial space as the global mammograms.

Typical outputs:

```text
data/preprocessed/
├── zoom_384x384/
├── roi_global_768x512/
└── dataset_index_zoom_384x384.csv
```

The local index stores the path to the corresponding global-space ROI mask. This mask is later used for Grad-CAM/ROI comparison without reopening the original DICOM files.

### Global preprocessing

```bash
python -m src.preprocessing.dataset_preprocessing --mode global
```

Typical outputs:

```text
data/preprocessed/
├── full_768x512/
└── dataset_index_full_768x512.csv
```

---

## Pairing local lesions with full mammograms

Fusion is performed at lesion level.

A local lesion is linked to its global mammogram using:

```text
patient_id
left or right breast
image view
```

The pairing implementation is located in:

```text
src/data/pairing.py
```

The global lookup is constrained so that a mammogram key maps to a single full-mammogram path, and the final merge is validated as `many_to_one`.

---

## Repository structure

```text
.
├── data/
│   ├── metadata/
│   ├── processed/
│   └── train_val_test_splits/
├── src/
│   ├── data/
│   │   ├── build_dataset.py
│   │   ├── pairing.py
│   │   ├── split_evaluation.py
│   │   ├── train_val_test_split.py
│   │   └── unique_patient_global.py
│   ├── preprocessing/
│   │   ├── dataset_preprocessing.py
│   │   └── dicom_handling.py
│   ├── modeling/
│   │   ├── fusion.py
│   │   ├── global_resnet50.py
│   │   └── local_resnet50.py
│   ├── training/
│   │   ├── dataset_preparation.py
│   │   ├── fusion_global_local_resnet50.py
│   │   ├── resnet50_global_branch.py
│   │   ├── resnet50_local_branch.py
│   │   ├── threshold_selection.py
│   │   └── training_utils.py
│   ├── evaluation/
│   │   ├── analyse_oof.py
│   │   ├── evaluation_utils.py
│   │   ├── generate_branch_oof.py
│   │   └── run_fusion_cv.py
│   ├── predict/
│   │   ├── evaluate_residual_threshold.py
│   │   └── evaluate_saved_model.py
│   ├── xAI_gradcam/
│   │   ├── grad_cam_residual_resnet.py
│   │   ├── grad_cam_resnet.py
│   │   └── gradcam_utils.py
│   ├── config.py
│   └── functions.py
├── draft_report.pdf
├── residual_threshold_seed_42.json
├── requirements.txt
└── README.md
```

Generated NumPy arrays, model checkpoints and plots are not intended to be version-controlled.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies include TensorFlow/Keras, NumPy, pandas, SciPy, scikit-learn, matplotlib, OpenCV, pydicom and pylibjpeg.

---

## Raw CBIS-DDSM location

By default, the project expects the raw DICOM hierarchy at:

```text
data/raw/cbis_ddsm/
```

To use another location:

```bash
export CBIS_DDSM_ROOT="/path/to/cbis_ddsm"
```

For example:

```bash
export CBIS_DDSM_ROOT="/home/user/datasets/cbis-ddsm/cbis_ddsm"
```

The path must point to the directory that directly contains folders such as:

```text
Mass-Training_P_00001_LEFT_CC/
Mass-Test_P_XXXXX_RIGHT_MLO/
...
```

---

## End-to-end workflow

### 1. Build the resolved dataset index

```bash
python -m src.data.build_dataset
```

### 2. Create or inspect patient-grouped splits

The exact splits used for the reported experiments are already stored under:

```text
data/train_val_test_splits/
```

They can be regenerated with:

```bash
python -m src.data.train_val_test_split
python -m src.data.unique_patient_global
```

Inspect the split distributions with:

```bash
python -m src.data.split_evaluation
```

### 3. Preprocess local and global inputs

```bash
python -m src.preprocessing.dataset_preprocessing --mode local
python -m src.preprocessing.dataset_preprocessing --mode global
```

### 4. Train the individual branches

```bash
python -m src.training.resnet50_local_branch
python -m src.training.resnet50_global_branch
```

### 5. Train fixed-split fusion models

Symmetric fusion:

```bash
python -m src.training.fusion_global_local_resnet50 --model symmetric
```

Residual fusion:

```bash
python -m src.training.fusion_global_local_resnet50 --model residual
```

### 6. Run five-fold fusion cross-validation

All outer folds:

```bash
python -m src.evaluation.run_fusion_cv --all
```

A single fold:

```bash
python -m src.evaluation.run_fusion_cv --fold 0
```

Use `--force` to retrain checkpoints that already exist.

### 7. Generate local/global OOF predictions

```bash
python -m src.evaluation.generate_branch_oof --all
```

Use `--overwrite` to overwrite existing branch OOF prediction files.

### 8. Analyse OOF predictions

```bash
python -m src.evaluation.analyse_oof
```

The analysis produces:

- fold-level metrics;
- pooled OOF metrics;
- paired fold differences;
- patient-cluster bootstrap distributions;
- 95% bootstrap confidence intervals;
- calibration tables;
- an OOF reliability diagram.

Cross-validation results are written under:

```text
src/models/fusion_cv_5fold_seed_42/
```

### 9. Select the residual operating threshold

```bash
python -m src.predict.evaluate_residual_threshold
```

This performs a validation threshold grid search and stores the selected operating point in:

```text
residual_threshold_seed_42.json
```

### 10. Evaluate saved models

Examples:

```bash
python -m src.predict.evaluate_saved_model --model local --scope native
python -m src.predict.evaluate_saved_model --model local --scope paired
python -m src.predict.evaluate_saved_model --model global --scope native
python -m src.predict.evaluate_saved_model --model symmetric --scope paired
python -m src.predict.evaluate_saved_model --model residual --scope paired
```

`paired` evaluation is useful when comparing a branch directly with the fusion models because it restricts evaluation to matched lesion/mammogram observations.

---

## Grad-CAM interpretability

The project adapts the Grad-CAM approach from the Keras computer-vision example.

### Individual branch Grad-CAM

Local:

```bash
python -m src.xAI_gradcam.grad_cam_resnet \
    --mode local \
    --idx 0 \
    --target_class predicted
```

Global:

```bash
python -m src.xAI_gradcam.grad_cam_resnet \
    --mode global \
    --idx 0 \
    --target_class predicted
```

`--target_class` accepts:

```text
predicted
0
1
```

### Residual-fusion Grad-CAM

```bash
python -m src.xAI_gradcam.grad_cam_residual_resnet \
    --idx 0 \
    --target_class predicted
```

The residual Grad-CAM pipeline can compare the global activation map with the lesion ROI in the same preprocessed global spatial space.

Reported ROI-related Grad-CAM measures include:

- fraction of Grad-CAM energy inside the ROI;
- whether the activation peak lies inside the ROI;
- Dice and IoU for the top 20% of activated pixels;
- ROI area fraction;
- activation enrichment relative to ROI area.

These measures are exploratory. Grad-CAM is treated as qualitative/diagnostic interpretability evidence and does not establish that the model learned clinically causal features.

---

## Reproducibility

The final configuration is centralised in:

```text
src/config.py
```

Main settings include:

```text
SEED = 42
BATCH_SIZE = 16
EPOCHS = 100

LOCAL_HEIGHT = 384
LOCAL_WIDTH = 384

GLOBAL_HEIGHT = 768
GLOBAL_WIDTH = 512

N_OUTER_FOLDS = 5
N_BOOTSTRAP = 10000
```

The repository version-controls:

- the original metadata;
- the resolved dataset index;
- the fixed patient-grouped train/validation/test splits;
- the source code;
- the reported residual operating-threshold JSON.

The following are intentionally external or generated:

- raw CBIS-DDSM DICOM files;
- preprocessed `.npy` arrays;
- trained model checkpoints;
- generated plots.

Deep-learning training can exhibit small run-to-run numerical variation depending on the TensorFlow execution environment. For this reason, the final architectural conclusions rely primarily on patient-grouped cross-validation, matched OOF predictions and paired patient-cluster bootstrap analysis rather than on a single training run.

---

## Main interpretation

The final experiments support three main observations:

1. **Lesion-centred information is more predictive than the full mammogram alone** for this CBIS-DDSM mass-classification task.
2. **Global context can add useful complementary information**, since both fusion strategies improve aggregate performance over the individual branches.
3. **Residual fusion is the strongest overall formulation in this study**, especially for BCE, average precision and calibration. Its ROC-AUC advantage over symmetric fusion is small and uncertain, so the results do not support claiming a clear ROC-ranking superiority between the two fusion designs.

The study therefore supports global-local modelling while suggesting that **how context is incorporated may matter as much as whether it is incorporated**.

---

## Limitations

Important limitations include:

- CBIS-DDSM is relatively small compared with modern screening datasets;
- only mass lesions are considered;
- performance varies substantially between patient partitions;
- the models operate on known abnormalities and therefore do not solve lesion detection;
- Grad-CAM localisation is not a direct measure of clinical validity;
- results should not be interpreted as evidence of real-world clinical performance.

See `draft_report.pdf` for the full methodology, literature review, development experiments, statistical analysis and discussion.

---

## License / academic use

This repository was developed as part of an academic machine-learning project.

CBIS-DDSM remains subject to the terms of its original data provider. No raw medical images are redistributed in this repository.
