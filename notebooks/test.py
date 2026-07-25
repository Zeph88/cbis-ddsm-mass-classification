import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
from src.config import DATASET_INDEX, IMAGES_ROOT, OUTPUT_NPY, SEED, BATCH_SIZE, EPOCHS, PIXELS_H, PIXELS_W
import math

df = pd.read_csv(OUTPUT_NPY / "test_split.csv")
print(df[df["label"]==1].count() / df.count())