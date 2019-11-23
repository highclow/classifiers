import logging
import torch
import torch.optim as optim

from .resnet import *

def get_net(net_name, num_classes, param_path, device="cpu"):
    logging.info("Create Network %s"%net_name)
    if net_name == "ResNet50":
      net = resnet50(num_classes=num_classes)
    elif net_name == "ResNet18":
      net = resnet18(num_classes=num_classes)
    else:
      logging.fatal("Please specify network!")
      sys.exit(1)

    if param_path:
      params = torch.load(param_path, map_location=device)
      for n in net.state_dict().keys():
        if n in params and params[n].shape == net.state_dict()[n].shape:
          net.state_dict()[n][...] = params[n]
    return net


def get_optimizer(net, cfgs):
    strategy = cfgs.get("train", "optimizer")
    lr = cfgs.getfloat("train", "base_lr")
    if strategy == "SGD":
      momentum = cfgs.getfloat("train", "momentum")
      weight_decay = cfgs.getfloat("train", "weight_decay")
      logging.info("Optimization strategy: %s"%strategy)
      logging.info("Base Learning Rate %f"%lr)
      logging.info("Momentum %f, Weight decay %f"%(momentum, weight_decay))
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
    else:
      logging.fatal("Please specify optimizer")
      sys.exit(1)
    return optimizer


def get_scheduler(optimizer, cfgs):
    policy = cfgs.get("train", "lr_policy")
    if policy == "multistep":
      milestones = list(map(int, cfgs.get("train", "step_size").split(",")))
      gamma = cfgs.getfloat("train", "gamma")
      logging.info("Scheduler policy %s with gamma %f"%(policy, gamma))
      scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    else:
      logging.fatal("Please specify schedular polciy")
      sys.exit(1)
    return scheduler
