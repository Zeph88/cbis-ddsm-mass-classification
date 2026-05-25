import pandas as pd
import numpy as np
import tensorflow as tf
from src.preprocessing.dicom_io import dicom_to_tf_tensor, apply_roi_mask
from src.config import DATASET_INDEX, IMAGES_ROOT, BATCH_SIZE
import math

def build_tf_dataset(df, batch_size=16, shuffle=True, seed=42):

    df = df[(df["keep"] == True)].reset_index(drop=True)

    paths = df["preprocessed_image_path"].tolist()
    labels = df["label"].astype("int32").tolist()

    def load_npy(path, label):
        image = np.load(path.decode("utf-8")).astype("float32")
        return image, label

    def tf_load_npy(path, label):
        image, label = tf.numpy_function(
            load_npy,
            [path, label],
            [tf.float32, tf.int32]
        )

        image.set_shape((224, 224, 1))
        label.set_shape(())

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(1)

    print(len(df), "steps:", math.ceil(len(df) / batch_size))

    return dataset

def main():
    train_ds = build_tf_dataset(
        DATASET_INDEX,
        IMAGES_ROOT,
        source="train",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    images, labels = next(iter(train_ds))

    return images, labels

if __name__ == "__main__":
    main()