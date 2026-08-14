# CBIS-DDSM Mass Classification

Deep-learning project for **benign vs malignant classification of known mammographic masses** using the CBIS-DDSM dataset.

The project investigates whether a lesion-centred representation can benefit from complementary information extracted from the full mammogram. The final implementation compares:

- a **local ResNet50 branch** trained on ROI-centred lesion crops;
- a **global ResNet50 branch** trained on full mammograms;
- a **residual local-global fusion model**, where the local prediction acts as the baseline and the global representation learns a contextual correction;
- **Grad-CAM** visualisations for qualitative interpretation.

## Scope

This project addresses **computer-aided diagnosis (CADx)** rather than lesion detection or screening.

The abnormality location is assumed to be known from the expert-provided CBIS-DDSM ROI mask. The task is therefore to classify a known mammographic mass as **benign** or **malignant**.

## Dataset

The project uses the **CBIS-DDSM mass subset**.

Raw DICOM files are **not distributed in this repository**. They must be obtained separately from the CBIS-DDSM collection before running the preprocessing pipeline.

The original metadata used by the project are version-controlled under:

```text
data/metadata/
├── mass_case_description_train_set.csv
└── mass_case_description_test_set.csv
```

The repository also contains the resolved dataset index:

```text
data/processed/mass_dataset_index.csv
```

The dataset-building code resolves the DICOM files using their metadata, including the DICOM `SeriesDescription` field.

### Raw data location

By default, the code expects the CBIS-DDSM DICOM hierarchy under:

```text
data/raw/cbis_ddsm/
```

A different location can be configured with the environment variable:

```bash
CBIS_DDSM_ROOT=/path/to/cbis_ddsm
```

Raw DICOM files and generated NumPy arrays are intentionally excluded from Git.

## Exact experimental splits

The exact train, validation and test partitions used for the reported experiments are version-controlled under:

```text
data/train_val_test_splits/
├── train_split.csv
├── val_split.csv
├── test_split.csv
├── train_split_global.csv
├── val_split_global.csv
└── test_split_global.csv
```

Splits are created at **patient level** to prevent patient leakage between train, validation and test sets.

The official CBIS-DDSM test partition is retained as the test set. A validation subset is derived from the official training data using grouped stratification.

For the global branch, multiple lesion rows belonging to the same mammogram are consolidated into one mammogram-level sample. If a mammogram contains both benign and malignant lesions, the malignant label is retained.

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
│   │   ├── train_val_test_split.py
│   │   ├── unique_patient_global.py
│   │   └── split_evaluation.py
│   ├── preprocessing/
│   │   ├── dataset_preprocessing.py
│   │   └── dicom_handling.py
│   ├── training/
│   │   ├── dataset_preparation.py
│   │   ├── resnet50_local_branch.py
│   │   ├── resnet50_global_branch.py
│   │   ├── fusion_global_local_resnet50.py
│   │   ├── threshold_selection.py
│   │   └── cnn_evaluation.py
│   ├── predict/
│   │   ├── evaluate_saved_model.py
│   │   ├── evaluate_residual_threshold.py
│   │   ├── resnet50_fusion.py
│   │   └── best_alpha.py
│   ├── xAI_gradcam/
│   │   ├── grad_cam_resnet.py
│   │   └── grad_cam_residual_resnet.py
│   ├── experiments/
│   │   └── ...
│   ├── config.py
│   └── functions.py
├── requirements.txt
└── README.md
```

`src/experiments/` contains intermediate experiments retained to document the development process. The final reported pipeline is implemented primarily under `src/preprocessing/`, `src/training/`, `src/predict/` and `src/xAI_gradcam/`.

## Pipeline

### 1. Build the dataset index

After downloading CBIS-DDSM and configuring `CBIS_DDSM_ROOT` if required:

```bash
python -m src.data.build_dataset
```

This resolves the relevant DICOM files and creates:

```text
data/processed/mass_dataset_index.csv
```

### 2. Build or inspect the data splits

The exact splits used in the reported experiments are already included in the repository.

They can be regenerated with:

```bash
python -m src.data.train_val_test_split
python -m src.data.unique_patient_global
```

Dataset distributions can be inspected with:

```bash
python -m src.data.split_evaluation
```

### 3. Preprocess the DICOM data

Preprocessing converts the DICOM inputs into reusable `.npy` arrays. This is performed separately from training so that DICOM decoding and image preprocessing do not need to be repeated for every experiment.

```bash
python -m src.preprocessing.dataset_preprocessing
```

Generated arrays are stored under `data/preprocessed/` and are not version-controlled.

### Local representation

The final local representation:

- uses the expert ROI mask to locate the lesion;
- calculates the lesion centroid from the mask;
- extracts a fixed-size lesion-centred crop;
- uses a final resolution of **384 × 384**.

### Global representation

The final global representation:

- uses the full mammogram;
- normalises breast orientation;
- removes isolated annotations and background artefacts using OpenCV thresholding, morphology and connected-component analysis;
- retains the main breast region;
- uses a final resolution of **768 × 512**.

## Final models

### Local ResNet50

The retained local model uses an ImageNet-pretrained ResNet50 backbone with frozen convolutional weights and a task-specific classification head.

```bash
python -m src.training.resnet50_local_branch
```

### Global ResNet50

The global model applies the same transfer-learning principle to the full mammogram representation.

```bash
python -m src.training.resnet50_global_branch
```

### Residual local-global fusion

The final fusion model treats the local branch as the primary classifier and learns a contextual correction from the combined local and global embeddings:

```text
final logit = local logit + contextual correction
```

The local and global branches are frozen during fusion training. The correction output is zero-initialised so that the fusion model initially reproduces the local prediction.

```bash
python -m src.training.fusion_global_local_resnet50
```

Generated model files are written to `src/models/`. This directory is created automatically when required and is excluded from Git.

## Threshold selection and evaluation

ROC-AUC and binary cross-entropy are used for model comparison, together with threshold-dependent metrics such as:

- accuracy;
- precision;
- sensitivity / recall;
- specificity;
- F1-score.

The final operating threshold is selected from **validation predictions only**, with sensitivity prioritised in the medical decision-support setting.

The retained rule selects a validation threshold providing at least **85% recall**.

Relevant evaluation scripts are located under:

```text
src/predict/
```

Plots are written to `src/plots/`. This directory is created automatically when required and is excluded from Git.

## Interpretability

Grad-CAM is used to inspect spatial evidence contributing to local and global predictions.

The final residual-fusion implementation is available in:

```bash
python -m src.xAI_gradcam.grad_cam_residual_resnet
```

Grad-CAM outputs are treated as qualitative interpretability evidence and not as proof that the model relies on clinically meaningful features.

## Installation

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies include:

- TensorFlow / Keras
- NumPy
- pandas
- SciPy
- scikit-learn
- matplotlib
- OpenCV
- pydicom
- pylibjpeg

## Reproducibility

The project uses fixed dataset partitions and fixed random seeds where applicable.

The principal experimental seeds are:

```text
42
123
999
```

Full reproduction requires obtaining the CBIS-DDSM DICOM dataset separately. The raw DICOM files are not included because they are external source data and are substantially larger than the source-code repository.

Generated `.npy` arrays, trained model checkpoints and plots are also excluded because they can be recreated from the raw data and the provided pipeline.

## Notes

This repository was developed as part of an academic machine-learning project. Intermediate experimental scripts are retained under `src/experiments/`, while the final pipeline is separated into dedicated data, preprocessing, training, evaluation and interpretability modules.
