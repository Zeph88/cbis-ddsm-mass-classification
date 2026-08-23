import tensorflow as tf
from src.evaluation.evaluation_utils import build_binary_metrics

def callbacks_for(checkpoint_path, log_path):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=False,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
        ),

        tf.keras.callbacks.CSVLogger(
            filename=str(
                log_path
            ),
        ),
    ]

def build_binary_metrics(thresholds=THRESHOLDS):
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]

    for threshold in thresholds:
        suffix = round(threshold * 100)

        metrics.extend([
            tf.keras.metrics.Recall(
                name=f"recall_{suffix}",
                thresholds=threshold,
            ),
            tf.keras.metrics.Precision(
                name=f"precision_{suffix}",
                thresholds=threshold,
            ),
        ])

    return metrics


def compile_binary_model(model, learning_rate=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=build_binary_metrics()
    )

    return model