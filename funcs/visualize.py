#coding:utf-8
import sys
import logging
import json
import collections
import numpy as np
from glob import glob
from sklearn import metrics
import matplotlib.font_manager
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

from utils import read_from_txt


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


def savefig(filename, fnr, tnr, threshold, fdr_limit, tnr_limit):
     index = (fnr < fdr_limit) * (tnr > tnr_limit)
     fnr = fnr[index]
     tnr = tnr[index]
     threshold = threshold[index]
     plt.xlim(0., fdr_limit)
     plt.ylim(tnr_limit, 1.)
     plt.plot(fnr, tnr, c='red', lw=1, alpha=0.3)
     plt.xlabel(u'拦截率')
     plt.ylabel(u'识别率')
     plt.savefig(filename)
     plt.close()


def visualize(cfgs):
    """ Find the failure case """
    filename = cfgs.get('visualize', 'filename')
    fdr_limit = cfgs.getfloat('visualize', 'fnr_limit')
    tnr_limit = cfgs.getfloat('visualize', 'tnr_limit')
    imagelist = glob(filename)
    if len(imagelist) > 0:
      for filename in imagelist:
        if '.npy' in filename:
          try:
            logging.info('Load file %s'%filename)
            data = np.load(filename)
          except:
            logging.fatal('Cannot load file %s'%filename)
            continue
          fnr, tnr, threshold = get_metric(data)
          save_file = filename.replace(".npy",".png")
          savefig(save_file, fnr, tnr, threshold, fdr_limit, tnr_limit)
        elif '.txt' in filename:
          imagelist, data = read_from_txt(filename)
          fnr, tnr, threshold = get_metric(data)
          save_file = filename.replace(".txt",".png")
          savefig(save_file, fnr, tnr, threshold, fdr_limit, tnr_limit)
          idx005 = np.abs(fnr - 0.005).argmin()
          idx01 = np.abs(fnr - 0.01).argmin()
          idx05 = np.abs(fnr - 0.05).argmin()
          th = threshold[idx01]
          logging.info("The threshold is %.4f when intercept rate is 0.01!"%th)
          res = {'tp': collections.defaultdict(list),
                 'fn': collections.defaultdict(list),
                 'tn': collections.defaultdict(list),
                 'fp': collections.defaultdict(list)}
          for path, item in zip(imagelist, data):
            key = path.split('/')[-2]
            if item[0] > th and item[-1] != 0:
               res['fp'][key].append((path, item[0]))
            elif item[0] < th and item[-1] == 0:
               res['fn'][key].append((path, item[0]))
#            elif item[0] > th and item[-1] == 0:
#               res['tp'][key].append((path, item[0]))
#            elif item[0] < th and item[-1] != 0:
#               res['tn'][key].append((path, item[0]))
          save_file = filename.replace('.txt', '.json')
          json.dump(res, open(save_file, 'w'), indent=2)
          
    else:
      logging.fatal("Please specify result name")
      sys.exit(1)
