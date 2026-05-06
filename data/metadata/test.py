from pathlib import Path
import pandas as pd

csv_path = Path("/home/julien/cbis-ddsm/data/metadata/mass_case_description_test_set.csv")
images_root = Path("/home/julien/datasets/cbis-ddsm/cbis_ddsm")

df = pd.read_csv(csv_path)

row = df.iloc[0]
print(row)
rel_path = Path(str(row["ROI mask file path"]).strip())

print("CSV relative path:")
print(rel_path)

current = images_root

for part in rel_path.parts:
    current = current / part
    print(current, "->", current.exists())