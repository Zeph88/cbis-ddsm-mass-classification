import pandas as pd
import numpy as np
import tensorflow as tf
from src.config import DATASET_INDEX, IMAGES_ROOT, BATCH_SIZE, PIXELS_H, PIXELS_W, SEED
import math


def build_tf_dataset(df, batch_size=16, shuffle=True, seed=42, path_image="preprocessed_image_path", added_path_image="", 
    image_height=PIXELS_H, image_width=PIXELS_W, added_image_height=0, added_image_width=0):

    has_added_image = added_path_image != ""
    paths = df[path_image].tolist()
    labels = df["label"].astype("int32").tolist()

    if has_added_image:
        added_paths = df[added_path_image].tolist()

    def load_npy(*args):
        if has_added_image:
            path, added_path, label = args
            image = np.load(path.decode("utf-8")).astype("float32")
            added_image = np.load(added_path.decode("utf-8")).astype("float32")

            return image, added_image, label

        path, label = args
        image = np.load(path.decode("utf-8")).astype("float32")

        return image, label

    def tf_load_npy(*args):
        if has_added_image:
            image, added_image, label = tf.numpy_function(load_npy, list(args), [tf.float32, tf.float32, tf.int32])
            image.set_shape((image_height, image_width, 1))
            added_image.set_shape((added_image_height, added_image_width, 1))
            label.set_shape(())

            return (image, added_image), label

        image, label = tf.numpy_function(load_npy, list(args), [tf.float32, tf.int32])
        image.set_shape((image_height, image_width, 1))
        label.set_shape(())

        return image, label

    if has_added_image:
        dataset = tf.data.Dataset.from_tensor_slices((paths, added_paths, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(tf_load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(1)

    print(len(df), "steps:", math.ceil(len(df) / batch_size))

    return dataset

def train_val_test_sets(data, batch_size=BATCH_SIZE, seed=SEED, path_image="preprocessed_image_path", added_path_image="", 
    image_height=PIXELS_H, image_width=PIXELS_W, added_image_height=0, added_image_width=0):

    train_ds = build_tf_dataset(data[data["set"]=="train"], batch_size=batch_size, shuffle=True, seed=seed, path_image=path_image, added_path_image=added_path_image, 
        image_height=image_height, image_width=image_width, added_image_height=added_image_height, added_image_width=added_image_width)

    val_ds = build_tf_dataset(data[data["set"]=="validation"], batch_size=batch_size, shuffle=False, seed=seed, path_image=path_image, added_path_image=added_path_image, 
        image_height=image_height, image_width=image_width, added_image_height=added_image_height, added_image_width=added_image_width)
        
    test_ds = build_tf_dataset(data[data["set"]=="test"], batch_size=batch_size, shuffle=False, seed=seed, path_image=path_image, added_path_image=added_path_image, 
        image_height=image_height, image_width=image_width, added_image_height=added_image_height, added_image_width=added_image_width)

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