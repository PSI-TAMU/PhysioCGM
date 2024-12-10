import os
import json
import numpy as np
from sklearn.metrics import roc_curve

def calculate_eer_threshold(preds, labels):
    fpr, tpr, thresholds = roc_curve(labels, preds)
    fnr = 1 - tpr  # False Negative Rate
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - fnr))]
    return eer_threshold

def calculate_confusion_matrix(preds, labels, threshold):
    preds = (preds > threshold).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    confusion_maxtrix = np.array([[tp, fp], [fn, tn]]) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return confusion_maxtrix, sensitivity, specificity


def load_eer_thresholds(subject, out_dir):
    ecg_results_path = os.path.join(out_dir, 'ecg', subject, 'results.json')
    ppg_results_path = os.path.join(out_dir, 'ppg', subject, 'results.json')
    eda_results_path = os.path.join(out_dir, 'eda', subject, 'results.json')

    with open(ecg_results_path, 'r') as f:
        ecg_results = json.load(f)
    with open(ppg_results_path, 'r') as f:
        ppg_results = json.load(f)
    with open(eda_results_path, 'r') as f:
        eda_results = json.load(f)
    eer_thresholds = {
        "ecg": ecg_results['eer_thresholds'],
        "ppg": ppg_results['eer_thresholds'],
        "eda": eda_results['eer_thresholds']
    }
    return eer_thresholds