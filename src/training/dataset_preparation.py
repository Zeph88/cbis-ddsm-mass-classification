import pandas as pd
import tensorflow as tf
from src.preprocessing.dicom_io import dicom_to_tf_tensor
from src.config import DATASET_INDEX, IMAGES_ROOT
import math

def build_tf_dataset(index_csv, images_root, source="train", batch_size=16, shuffle=True):
    df = pd.read_csv(index_csv)

    df = df[
        (df["keep"] == True) &
        (df["source"] == source)
    ].reset_index(drop=True)

    paths = df["resolved_crop_rel_path"].tolist()
    labels = df["label"].astype("int32").tolist()

    def generator():
        for rel_path, label in zip(paths, labels):
            dicom_path = images_root / rel_path
            image = dicom_to_tf_tensor(dicom_path)
            yield image, label

    dataset = tf.data.Dataset.from_generator(
        lambda: generator(),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps = math.ceil(len(df) / batch_size)
    print(steps)
    print(source, len(df))

    return dataset

def main():
    train_ds = build_tf_dataset(
        DATASET_INDEX,
        IMAGES_ROOT,
        source="train",
        batch_size=16,
        shuffle=True
    )

    images, labels = next(iter(train_ds))

    return images, labels

if __name__ == "__main__":
    main()