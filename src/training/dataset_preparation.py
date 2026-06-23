import pandas as pd
import numpy as np
import tensorflow as tf
from src.config import DATASET_INDEX, IMAGES_ROOT, BATCH_SIZE, PIXELS_H, PIXELS_W, SEED
import math


def build_tf_dataset(df, batch_size=16, shuffle=True, seed=42):

    # df = df[(df["keep"] == True)].reset_index(drop=True)

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

        image.set_shape((PIXELS_H, PIXELS_W, 1))
        label.set_shape(())

        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(1)

    print(len(df), "steps:", math.ceil(len(df) / batch_size))

    return dataset


def train_val_test_sets(
        data,
        batch_size=BATCH_SIZE, 
        seed=SEED
        ):

    train_ds = build_tf_dataset(
        data[data["set"]=="train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )

    val_ds = build_tf_dataset(
        data[data["set"]=="validation"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    test_ds = build_tf_dataset(
        data[data["set"]=="test"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )

    return train_ds, val_ds, test_ds


def cnn_steps(data):

    train_size = len(data[data["set"]=="train"])
    val_size = len(data[data["set"]=="validation"])
    test_size = len(data[data["set"]=="test"])

    train_steps = math.ceil(train_size / BATCH_SIZE)
    val_steps = math.ceil(val_size / BATCH_SIZE)
    test_steps = math.ceil(test_size / BATCH_SIZE)

    print(f"Train: {train_size} samples, {train_steps} steps")
    print(f"Val: {val_size} samples, {val_steps} steps")
    print(f"Test: {test_size} samples, {test_steps} steps")

    return train_steps, val_steps, test_steps

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