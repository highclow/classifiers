#coding:utf-8
import sys
import logging
import numpy as np
from glob import glob
from sklearn import metrics
import matplotlib.font_manager
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

from utils import write_to_txt


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

    false_neg_rate = []  # 
    true_neg_rate = []
    threshold = []
    for k, idx in enumerate(true_indices):
      false_neg_rate.append(k * 1. / positive)
      true_neg_rate.append((idx - 1 - k) * 1. / negative)
      threshold.append(scores[idx-1])

    return np.array(false_neg_rate), np.array(true_neg_rate), np.array(threshold)


def visualize(cfgs):
    """ Find the failure case """
    filename = cfgs.get('visualize', 'filename')
    fdr_limit = cfgs.getfloat('visualize', 'fnr_limit')
    tnr_limit = cfgs.getfloat('visualize', 'tnr_limit')
    imagelist = glob(filename)
    if len(imagelist) > 0:
      for filename in imagelist:
        try:
          logging.info('Load file %s'%filename)
          if '.npy' in filename:
            data = np.load(filename)
          elif '.txt' in filename:

        except:
          logging.fatal('Cannot load file %s'%filename)
          continue
        fnr, tnr, threshold = get_metric(data)
        index = (fnr < fdr_limit) * (tnr > tnr_limit)
        fnr = fnr[index]
        tnr = tnr[index]
        threshold = threshold[index]
        plt.xlim(0., 0.1)
        plt.ylim(0.9, 1.)
        plt.plot(fnr, tnr, c='red', lw=1, alpha=0.3)
        plt.xlabel(u'拦截率')
        plt.ylabel(u'识别率')
        plt.savefig(filename.replace(".npy",".png"))
        plt.close()
    else:
      logging.fatal("Please specify result name")
      sys.exit(1)
