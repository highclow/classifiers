import os, sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from datasets import get_imagelist_dataloader
from models import get_net, get_optimizer, get_scheduler


def val_net(net, dataloader, model_path, cfgs):
    logging.info("Start validating network...")
    device = cfgs.get("val", "device")
    display = cfgs.getint("val", "display")

    mlog = utils.MetricLogger(delimiter="  ")
    net.eval()
    net = net.to(device)
    with torch.no_grad():
      res = []
      mlog.clear()
      for inputs, labels in mlog.log_every(dataloader, display, "val"):
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
        mlog.meters["acc1"].update(acc1.item(), n=batch_size)
      mlog.synchronize_between_processes()
      logging.info(" * Acc@1 {top1.global_avg:.3f}".format(top1=mlog.acc1))
      res_path = model_path.replace(
             ".pt", "_{top1.global_avg:.3f}.npy".format(top1=mlog.acc1))
      logging.info(" * Writing results to %s"%res_path)
      np.save(res_path, np.concatenate(res))


def validate(cfgs):
    utils.set_device(cfgs.get("val", "device"),
                     cfgs.getint("val", "device_id"))
    loader = get_imagelist_dataloader(cfgs, "val")
    if cfgs.get("val", "params"):
      model_path = cfgs.get("val", "params")
      logging.info("Loading Model %s"%model_path)
      net = get_net(cfgs.get("model", "net"),
                    cfgs.getint("model", "classes"),
                    model_path)
      val_net(net, loader, model_path, cfgs)
    else:
      path = os.path.join(cfgs.get("train", "snapshot_prefix"),
                          cfgs.get("model", "net"))
      model_list = [p for p in os.listdir(path) if ".pt" in p]
      for d in sorted(sorted(model_list), key=len):
        model_path = os.path.join(path, d)
        logging.info("Loading Model %s"%model_path)
        net = get_net(cfgs.get("model", "net"),
                      cfgs.getint("model", "classes"),
                      model_path)
        val_net(net, loader, model_path, cfgs)

