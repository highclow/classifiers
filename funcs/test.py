import os, sys
import logging
import numpy as np
from PIL import Image
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from datasets import default_list_reader, default_transform
from models import get_net


def test_net(net, model_path, cfgs):
    logging.info("Start testing network...")
    device = cfgs.get("test", "device")
    display = cfgs.getint("test", "display")
    root = cfgs.get("test", "root")
    filelist = cfgs.get("test", "imagelist")

    mlog = utils.MetricLogger(delimiter="  ")
    net.eval()
    net = net.to(device)
    with torch.no_grad():
      res = []
      mlog.clear()
      imagelist = default_list_reader(filelist)
      transform = default_transform()
      for imgpath, label in mlog.log_every(imagelist, display, "test"):
        img = Image.open(os.path.join(root, imgpath))
        inputs = torch.as_tensor(transform(img)[np.newaxis,:])
        inputs = inputs.to(device)

        # forward 
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        res.append([imgpath] + outputs.cpu().numpy().tolist()[0] + [label])

      mlog.synchronize_between_processes()
      res_path = model_path.replace(".pt", "_%s"%filelist.split('/')[-1])
      logging.info(" * Writing results to %s"%res_path)
      utils.write_to_txt(res_path, res)


def test(cfgs):
    utils.set_device(cfgs.get("test", "device"),
                     cfgs.getint("test", "device_id"))
    model_str = cfgs.get("test", "params")
    if model_str:
      model_list = glob(model_str)
    else:
      logging.info("Please specify the model!")
      sys.exit(1)

    for model_path in model_list:
      logging.info("Loading model %s"%model_path)
      net = get_net(cfgs.get("model", "net"),
                    cfgs.getint("model", "classes"),
                    model_path)
      test_net(net, model_path, cfgs)
