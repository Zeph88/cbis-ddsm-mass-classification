import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from src.config import OUTPUT_NPY, SEED, PROCESSED_DIR

IMAGES_INDEX = OUTPUT_NPY / "soft_mask"
# PREPROCESSED_INDEX = OUTPUT_NPY / "dataset_index_roi_masked.csv"
PREPROCESSED_INDEX = PROCESSED_DIR / "mass_dataset_index.csv"

df = pd.read_csv(PREPROCESSED_INDEX)

def check_no_patient_leakage(train_df, val_df, test_df=None):
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    train_val_overlap = train_patients & val_patients

    print("train ∩ val patients:", len(train_val_overlap))

    if train_val_overlap:
        raise ValueError(f"Patient leakage between train and val: {train_val_overlap}")

    if test_df is not None:
        test_patients = set(test_df["patient_id"])

        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients

        print("train ∩ test patients:", len(train_test_overlap))
        print("val ∩ test patients:", len(val_test_overlap))

        if train_test_overlap:
            raise ValueError(f"Patient leakage between train and test: {train_test_overlap}")

        if val_test_overlap:
            raise ValueError(f"Patient leakage between val and test: {val_test_overlap}")
    
    return len(train_val_overlap), len(train_test_overlap), len(val_test_overlap)

def train_test_split_no_leakage(df, train_prc=0.8, seed=SEED):
    splitrule = GroupShuffleSplit(
        n_splits = 1,
        train_size=train_prc,
        random_state = seed
    )

    train_index, val_index = next(
        splitrule.split(
            df,
            df["label"],
            groups = df["patient_id"]
        )
    )

    train_df = df.iloc[train_index].copy()
    val_df = df.iloc[val_index].copy()

    return train_df , val_df

train_df = df[
    (df["keep"] == True) &
    (df["source"] == "train")
].reset_index(drop=True)

test_df = df[
    (df["keep"] == True) &
    (df["source"] == "test")
].reset_index(drop=True)

train_df, val_df = train_test_split_no_leakage(
    train_df,
    train_prc=0.8,
    seed=SEED
)

train_val_overlap, train_test_overlap, val_test_overlap = check_no_patient_leakage(train_df, val_df, test_df)

if train_val_overlap==0 and train_test_overlap==0 and val_test_overlap==0:
    train_df.to_csv(OUTPUT_NPY / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_NPY / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_NPY / "test_split.csv", index=False)