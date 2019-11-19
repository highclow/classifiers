#coding:utf-8
import os, sys
import cv2
import logging
import json
import collections
import numpy as np
from glob import glob
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.font_manager
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
    result_list = sorted(glob(filename))
    if len(result_list) > 0:
      for filename in result_list:
        if '.npy' in filename:
          try:
            logging.info('Load file %s'%filename)
            data = np.load(filename)
            res_file = 'val_list' if 'val_list' in filename else 'test_list'
            imagelist = os.path.join(cfgs.get('visualize', 'prefix'),
                                              '%s.txt'%res_file)
            imagelist, _ = read_from_txt(imagelist)
            assert(len(data)==len(imagelist))
          except:
            logging.fatal('Cannot load file %s'%filename)
            continue
          fnr, tnr, threshold = get_metric(data)
          save_base = filename.replace(".npy","")
        elif '.txt' in filename:
          imagelist, data = read_from_txt(filename)
          save_base = filename.replace(".txt","")

        fnr, tnr, threshold = get_metric(data)
        savefig(save_base+'.png', fnr, tnr, threshold, fdr_limit, tnr_limit)

        idx005 = np.abs(fnr - 0.005).argmin()
        idx01 = np.abs(fnr - 0.01).argmin()
        idx05 = np.abs(fnr - 0.05).argmin()
        th = threshold[idx01]
        logging.info("Threshold: %.4f, intercept rate: %.4f, recognize rate: %.4f!"%(
                     th,fnr[idx01],tnr[idx01]))
        res = {'fpv': collections.defaultdict(list),
               'fnv': collections.defaultdict(list),
               #'tpv': collections.defaultdict(list),
               #'tnv': collections.defaultdict(list),
               #'tpt': collections.defaultdict(list),
               #'tnt': collections.defaultdict(list),
               'fnt': collections.defaultdict(list),
               'fpt': collections.defaultdict(list)}
        for path, item in zip(imagelist, data):
          key = path.split('/')[-2]
          if item[0] > th and item[-1] != 0:
             res['fpv'][key].append((path, item[0]))
          elif item[0] < th and item[-1] == 0:
             res['fnv'][key].append((path, item[0]))
#          elif item[0] > th and item[-1] == 0:
#             res['tp'][key].append((path, item[0]))
#          elif item[0] < th and item[-1] != 0:
#             res['tn'][key].append((path, item[0]))

        if 'val_list' in filename:
          try:
            test_res_file = glob(filename.split('val')[0] + 'test*')[0]
            imagelist = os.path.join(cfgs.get('visualize', 'prefix'),
                                              'test_list.txt')
            imagelist, _ = read_from_txt(imagelist)
            data = np.load(test_res_file)
            fnr, tnr, threshold = get_metric(data)
            idx = np.abs(fnr - th).argmin()
            logging.info("Test Threshold: %.4f, intercept rate: %.4f, recognize rate: %.4f!"%(
                         th,fnr[idx01],tnr[idx01]))
            for path, item in zip(imagelist, data):
              key = path.split('/')[-2]
              if item[0] > th and item[-1] != 0:
                 res['fpt'][key].append((path, item[0]))
              elif item[0] < th and item[-1] == 0:
                 res['fnt'][key].append((path, item[0]))
          except:
            continue
        json.dump(res, open(save_base+'.json', 'w'), indent=2)

        if cfgs.getboolean('visualize','samples'):
          root = cfgs.get('visualize', 'root')
          nrows = cfgs.getint('visualize', 'nrows')
          size = cfgs.getint('visualize', 'size')
          for key, item in res.items():
            res_img = []
            for k, v in item.items():
              print(key, k, len(v))
              col = 0
              save_img = np.ones((size, size*nrows, 3)) * 255
              for d in v:
                img = cv2.imread(os.path.join(root, d[0]))
                img = cv2.resize(img, (size,size))
                save_img[:, (col)*size:(col+1)*size, :] = np.copy(img)
                col += 1
                if col == nrows:
                  cv2.putText(save_img,k, (0,size), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (255,0,0), 2)
                  res_img.append(save_img.copy())
                  save_img = np.ones((size, size*nrows, 3)) * 255
                  col = 0

              if col != 0:
                cv2.putText(save_img,k, (0,size), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2)
                res_img.append(save_img.copy())
            
            plt.imsave('%s_%s.png'%(save_base, key), np.vstack(res_img).astype('uint8'))
         
    else:
      logging.fatal("Please specify result name")
      sys.exit(1)
