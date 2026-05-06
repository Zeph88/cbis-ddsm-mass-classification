from pathlib import Path 
import pandas as pd 
import numpy as np 
from src.functions import fetch_image, check_path
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT

df_train, df_test, images_root = fetch_image(TRAIN_CSV, TEST_CSV, IMAGES_ROOT)

_MAPPING_PATHOLOGY = {
    'BENIGN' : 0,
    'BENIGN_WITHOUT_CALLBACK' : 0,
    'MALIGNANT' : 1
}

df_train["pathology_id"] = df_train["pathology"].map(_MAPPING_PATHOLOGY)

for idx, row in df_train.iterrows(): 
    isfolder, path_to_folder = check_path(row["ROI mask file path"], images_root) 
    if isfolder: 
        print(f"patient: {row['patient_id']}, nb: {len(list(path_to_folder.glob('*.dcm')))}, pathology : {row['pathology_id']}")