from pathlib import Path

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
PIXELS_H = 384
PIXELS_W = 384
LOCAL_HEIGHT = 384
LOCAL_WIDTH = 384
GLOBAL_HEIGHT = 768
GLOBAL_WIDTH = 512
CROP_SIZE = 512 # Liao methodology

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

IMAGES_ROOT = Path(
    os.environ.get(
        "CBIS_DDSM_ROOT",
        PROJECT_ROOT / "data" / "raw" / "cbis_ddsm"
    )
)

SRC_ROOT = PROJECT_ROOT / "src"
MODEL_ROOT = SRC_ROOT / "models"
METADATA_DIR = DATA_ROOT / "metadata"
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_NPY = DATA_ROOT / "preprocessed"
OUTPUT_MODEL = SRC_ROOT / "models"
OUTPUT_PLOT = SRC_ROOT / "plots"

TRAIN_CSV = METADATA_DIR / "mass_case_description_train_set.csv"
TEST_CSV = METADATA_DIR / "mass_case_description_test_set.csv"

DATASET_INDEX = PROCESSED_DIR / "mass_dataset_index.csv"