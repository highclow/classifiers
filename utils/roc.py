import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def get_roc(vec, cls=0):
    """ the format is an array with size N * (C+1).
        N is num of samples,
        C is num of classes, 
        The last column is the groundtruth start from 0 """
    N = vec.shape[0]
    scores = vec[:,0]
    labels = (vec[:,-1] == cls).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    return fpr, tpr, thresholds


def get_metric(vec, cls=0):
    N = vec.shape[0]
    scores = vec[:,0]
    labels = (vec[:,-1] == cls).astype(int)
    sorted_idx = scores.argsort()
    scores = scores[sorted_idx]
    labels = labels[sorted_idx]

    true_indices = np.where(labels == 1)[0]
    positive = len(true_indices)
    negative = N - positive

    intercept_rate = []
    recognition_rate = []
    threshold = []
    for k, idx in enumerate(true_indices):
      intercept_rate.append(k * 1. / positive)
      recognition_rate.append((idx - 1 - k) * 1. / negative)
      threshold.append(scores[idx-1])

    return np.array(intercept_rate), np.array(recognition_rate), np.array(threshold)

