import numpy as np
from sklearn.metrics import roc_auc_score
from src.config import TRAIN_CSV, TEST_CSV, IMAGES_ROOT, OUTPUT_NPY, BATCH_SIZE, SEED

def cnn_predict(model, test_dataset):
    y_true = []
    y_prob = []

    for images, labels in test_dataset:
        probs = model(images, training=False).numpy().ravel()
        y_prob.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    print(f"minimum probability : {y_prob.min()}, maximum probability : {y_prob.max()}, average probability : {y_prob.mean()}")

    return y_prob, y_true

def evaluate_thresholds(y_prob, y_true, thresholds=[0.35, 0.40, 0.45, 0.50]):
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc}")

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        accuracy = (tp + tn) / len(y_true)

        print(f"threshold : {threshold}, accuracy : {accuracy}, precision : {precision}, recall : {recall}")