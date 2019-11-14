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
