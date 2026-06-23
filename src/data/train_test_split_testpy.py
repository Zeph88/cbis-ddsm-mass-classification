import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.config import OUTPUT_NPY, SEED, PROCESSED_DIR

DATASET_INDEX = PROCESSED_DIR / "mass_dataset_index_test.csv"

df = pd.read_csv(DATASET_INDEX)

def check_no_patient_leakage(train_df, test_df=None):
    train_patients = set(train_df["patient_id"])
    
    if test_df is not None:
        test_patients = set(test_df["patient_id"])
        train_test_overlap = train_patients & test_patients
        
        print("train ∩ test patients:", len(train_test_overlap))

        if train_test_overlap:
            raise ValueError(f"Patient leakage between train and test: {train_test_overlap}")

    return len(train_test_overlap)

train_df = df[
    (df["keep"] == True) &
    (df["source"] == "train")
].reset_index(drop=True)

test_df = df[
    (df["keep"] == True) &
    (df["source"] == "test")
].reset_index(drop=True)

train_test_overlap = check_no_patient_leakage(train_df, test_df)

if train_test_overlap==0:
    train_df.to_csv(OUTPUT_NPY / "train_split_test.csv", index=False)
    test_df.to_csv(OUTPUT_NPY / "test_split_test.csv", index=False)