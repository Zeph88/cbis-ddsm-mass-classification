import sys
from pathlib import Path 
import pandas as pd 
import numpy as np 

PROJECT_ROOT = Path("/home/julien/cbis-ddsm")
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT

def fetch_image(path_csv_train, path_csv_test, image_root):
    csv_path_train = Path(path_csv_train)
    csv_path_test = Path(path_csv_test)
    images_root = Path(image_root)

    df_train = pd.read_csv(csv_path_train)
    df_test = pd.read_csv(csv_path_test)

    return df_train, df_test, images_root

df_train, df_test, images_root = fetch_image(TRAIN_CSV, TEST_CSV, IMAGES_ROOT)

df_train["source"] = 'train' 
df_test["source"] = 'test' 
frames = [df_train, df_test] 
df = pd.concat(frames) 
df_unique = np.array([df[["patient_id", "source"]]]) 

patient_set = {} 
for row in df_unique[0]: 
    if row[0] in patient_set.keys(): 
        if row[1] in patient_set[row[0]].keys(): 
            patient_set[row[0]][row[1]] += 1 
        else: 
            patient_set[row[0]][row[1]] = 1 
    else:
        patient_set[row[0]] = {} 
        patient_set[row[0]][row[1]] = 1 

exceptions = [x for x in patient_set if len(patient_set[x]) > 1] 
print(exceptions)

# print(df_train["patient_id"].value_counts().describe())
# print(df_test["patient_id"].value_counts().describe())

# print(df_train["patient_id"].value_counts().head(20))
# print(df_test["patient_id"].value_counts().head(20))

df_test