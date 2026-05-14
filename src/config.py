from pathlib import Path

SEED = 42

PROJECT_ROOT = Path("/home/julien/cbis-ddsm")
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_DIR = DATA_ROOT / "metadata"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_NPY = DATA_ROOT / "preprocessed"

IMAGES_ROOT = Path("/home/julien/datasets/cbis-ddsm/cbis_ddsm")

TRAIN_CSV = METADATA_DIR / "mass_case_description_train_set.csv"
TEST_CSV = METADATA_DIR / "mass_case_description_test_set.csv"

DATASET_INDEX = PROCESSED_DIR / "mass_dataset_index.csv"