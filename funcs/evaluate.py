import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from datasets import get_imagelist_dataloader
from models import get_net, get_optimizer, get_scheduler


def eval_net(net, dataloader, model_path, cfgs):
    logging.info('Start evaluating network...')
    device = cfgs.get('eval', 'device')
    display = cfgs.getint('eval', 'display')

    mlog = utils.MetricLogger(delimiter="  ")
    net.eval()
    net = net.to(device)
    with torch.no_grad():
      res = []
      mlog.clear()
      for inputs, labels in mlog.log_every(dataloader, display, 'eval'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward 
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)

        res.append(np.hstack((outputs.cpu().numpy(),
                              labels.cpu().numpy()[:,np.newaxis])))
        # print statistics
        acc1, = utils.accuracy(outputs, labels, topk=(1,))
        batch_size = inputs.shape[0]
        mlog.meters['acc1'].update(acc1.item(), n=batch_size)
      mlog.synchronize_between_processes()
      logging.info(' * Acc@1 {top1.global_avg:.3f}'.format(top1=mlog.acc1))
      res_path = model_path.replace(
             '.pt', '_{top1.global_avg:.3f}.npy'.format(top1=mlog.acc1))
      logging.info(' * Writing results to %s'%res_path)
      np.save(res_path, np.concatenate(res))


def evaluate(cfgs):
    utils.set_device(cfgs.get('eval', 'device'),
                     cfgs.getint('eval', 'device_id'))
    loader = get_imagelist_dataloader(cfgs, 'eval')
    if cfgs.get('eval', 'params'):
      model_path = cfgs.get('eval', 'params')
      logging.info('Loading Model %s'%model_path)
      net = get_net(cfgs.get('model', 'net'),
                    cfgs.getint('model', 'classes'),
                    model_path)
      eval_net(net, loader, model_path, cfgs)
    else:
      path = os.path.join(cfgs.get('train', 'snapshot_prefix'),
                          cfgs.get('model', 'net'))
      for d in sorted(sorted(os.listdir(path)), key=len):
        model_path = os.path.join(path, d)
        logging.info('Loading Model %s'%model_path)
        net = get_net(cfgs.get('model', 'net'),
                      cfgs.getint('model', 'classes'),
                      model_path)
        eval_net(net, loader, model_path, cfgs)

