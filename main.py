import os
import argparse
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F

import logging
import utils
from configs import get_config
from funcs import train, evaluate


def get_command():
    parser = argparse.ArgumentParser(description=
                                     'Process for training classifier')
    parser.add_argument('func', choices=func_map.keys(),
                        help='functions. train/eval/test')
    parser.add_argument('-f','--config', type=str, required=True,
                        help='configure file')
    parser.add_argument('--weight', type=str, default=None,
                        help='weight for initialization or evaluation')
    parser.add_argument('-l','--outlog', type=str,
                        help='output log file')
    args = parser.parse_args()
    return args


def test_net(cfgs, test_model):
    if device == 'cuda':
      device_id = cfgs.getint('model', 'device_id')
      torch.cuda.set_device(device_id)

    logging.info('Create Data Loader')
    testloader = get_imagelist_dataloader(cfgs, 'test')

    logging.info('Create Network')
    path = os.path.join(cfgs.get('train', 'snapshot_prefix'),
                        cfgs.get('model', 'net'))

    mlog = utils.MetricLogger(delimiter="  ")
    display = cfgs.getint('test', 'display')
    with torch.no_grad():
      if test_model is not None:
          logging.info('Current Model: %s'%test_model)
          net = get_net(cfgs.get('model', 'net'), test_model, 3, device)
          net.eval()
          net.to(device)

          res = []
          mlog.clear()
          for inputs, labels in mlog.log_every(testloader, display, 'test'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward 
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)

            res.append(np.hstack((outputs.cpu().numpy(), labels.cpu().numpy()[:,np.newaxis])))
            # print statistics
            acc1, = utils.accuracy(outputs, labels, topk=(1,))
            batch_size = inputs.shape[0]
            mlog.meters['acc1'].update(acc1.item(), n=batch_size)
          mlog.synchronize_between_processes()
          logging.info(' * Acc@1 {top1.global_avg:.3f}'.format(top1=mlog.acc1))
          res_path = test_model.replace('.pt', '_{top1.global_avg:.3f}.npy'.format(top1=mlog.acc1))
          logging.info(' * Writing results to %s'%res_path)
          np.save(res_path, np.concatenate(res))
      else:
        for d in sorted(sorted(os.listdir(path)), key=len):
          model_path = os.path.join(path, d)
          res_path = model_path.replace('.pt', '_*.npy')
          logging.info('Current Model: %s'%model_path)
          net = get_net(cfgs.get('model', 'net'), model_path, 3, device)
          net.eval()
          net.to(device)

          res_files = glob(res_path)
          if len(res_files) != 0:
              logging.info('Current Model: %s Has Results %s!'%(model_path,res_files[0]))
              continue
          logging.info('Current Model: %s'%model_path)
          net = get_net(cfgs.get('model', 'net'), model_path, 3, device)
          net.eval()
          net.to(device)
  
          res = []
          mlog.clear()
          for inputs, labels in mlog.log_every(testloader, display, 'test'):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # forward 
            outputs = net(inputs)
            outputs = F.softmax(outputs, dim=1)
    
            res.append(np.hstack((outputs.cpu().numpy(), labels.cpu().numpy()[:,np.newaxis])))
            # print statistics
            acc1, = utils.accuracy(outputs, labels, topk=(1,))
            batch_size = inputs.shape[0]
            mlog.meters['acc1'].update(acc1.item(), n=batch_size)
          mlog.synchronize_between_processes()
          logging.info(' * Acc@1 {top1.global_avg:.3f}'.format(top1=mlog.acc1))
          res_path = os.path.join(path, d.replace('.pt', '_{top1.global_avg:.3f}.npy'.format(top1=mlog.acc1)))
          logging.info(' * Writing results to %s'%res_path)
          np.save(res_path, np.concatenate(res))


def test(cfgs, eights):
    pass


func_map = {'train' : train,
            'eval': evaluate,
            'test' :  test}


if __name__ == '__main__':
    args = get_command()
    utils.set_log(args.outlog)
    logging.info(args)

    cfgs = get_config(args)

    logging.info('Start %s the network'%args.func)
    func_map[args.func](cfgs, args.weight)
