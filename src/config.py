from pathlib import Path

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
PIXELS_H = 256
PIXELS_W = 256
CROP_SIZE = 512 # Liao methodology

PROJECT_ROOT = Path("/home/julien/cbis-ddsm")
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_ROOT = SRC_ROOT / "models"
METADATA_DIR = DATA_ROOT / "metadata"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_NPY = DATA_ROOT / "preprocessed"
OUTPUT_MODEL = SRC_ROOT / "models"

IMAGES_ROOT = Path("/home/julien/datasets/cbis-ddsm/cbis_ddsm")

TRAIN_CSV = METADATA_DIR / "mass_case_description_train_set.csv"
TEST_CSV = METADATA_DIR / "mass_case_description_test_set.csv"

DATASET_INDEX = PROCESSED_DIR / "mass_dataset_index.csv"