import os
import pathlib
from pathlib import Path
import pandas as pd 
import numpy as np 
from src.functions import fetch_image, find_dicom_by_series_description
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT

df_train, df_test, images_root = fetch_image(TRAIN_CSV, TEST_CSV, IMAGES_ROOT)

_MAPPING_PATHOLOGY = {
    'BENIGN' : 0,
    'BENIGN_WITHOUT_CALLBACK' : 0,
    'MALIGNANT' : 1
}

frame = [df_train, df_test]
df = pd.concat(frame)

df["pathology_id"] = df["pathology"].map(_MAPPING_PATHOLOGY)

df["resolved_roi_rel_path"] = df["ROI mask file path"].apply(
    lambda p: find_dicom_by_series_description(
        p,
        IMAGES_ROOT,
        "ROI mask images"
    )
)

df["resolved_crop_rel_path"] = df["cropped image file path"].apply(
    lambda p: find_dicom_by_series_description(
        p,
        IMAGES_ROOT,
        "cropped images"
    )
)

print(df['resolved_roi_rel_path'])
