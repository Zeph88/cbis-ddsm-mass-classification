from src.config import OUTPUT_NPY
import pandas as pd

df = pd.read_csv(OUTPUT_NPY / "dataset_index_zoom_256x256.csv")
df = df.groupby("lesion_key").size()
df = df.reset_index()
print(df[df[0]==1])