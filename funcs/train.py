import sys
import logging
import torch
import torch.nn as nn
import utils
from datasets import get_imagelist_dataloader
from models import get_net, get_optimizer, get_scheduler


def train_net(net, criterion, dataloader, optimizer, scheduler, cfgs):
    logging.info('Start training network...')
    lr_decay_mode = cfgs.get('train', 'lr_decay_mode')
    device = cfgs.get('train', 'device')
    display = cfgs.getint('train', 'display')
    snapshot = cfgs.getint('train', 'snapshot')

    mlog = utils.MetricLogger(delimiter="  ")
    mlog.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    net.train()
    net = net.to(device)
    iters = 0
    if lr_decay_mode == 'iter':
      while True:
        mlog.clear()
        for inputs, labels in mlog.log_every(dataloader, display, 'train'):
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          scheduler.step()

          # print statistics
          acc1, = utils.accuracy(outputs, labels, topk=(1,))
          batch_size = inputs.shape[0]
          mlog.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
          mlog.meters['acc1'].update(acc1.item(), n=batch_size)
          iters += 1

          if iters % snapshot == 0:
              path = os.path.join(cfgs.get('train', 'snapshot_prefix'),
                                  cfgs.get('model', 'net'))
              utils.mkdir(path)
              torch.save(net.state_dict(), path + '/' + 'iter_%d.pt'%iters)
    elif lr_decay_mode == 'epoch':
        logging.fatal("Epoch decay is not implentation")
        sys.exit(1)
    else:
        logging.fatal("Please specific lr_decay_mode. " +
                      "Currently only iter and epoch are accepted!")
        sys.exit(1)

    logging.info('Finished Training')


def train(cfgs):
    utils.set_device(cfgs.get('train', 'device'),
                     cfgs.getint('train', 'device_id'))
    trainloader = get_imagelist_dataloader(cfgs, 'train')

    net = get_net(cfgs.get('model', 'net'),
                  cfgs.getint('model', 'classes'),
                  cfgs.get('train', 'params'))
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, cfgs)
    scheduler = get_scheduler(optimizer, cfgs)
    train_net(net, criterion, trainloader, optimizer, scheduler, cfgs)


###################
# Adversarial Train, Still Debugging
###################
# FGSM
# PGD
def adversarial_train(net, criterion, dataloader, optimizer, cfgs):
    lr_decay_mode = cfgs.get('train', 'lr_decay_mode')
    device = cfgs.get('model', 'device')
    display = cfgs.getint('train', 'display')
    snapshot = cfgs.getint('train', 'snapshot')

    mlog = utils.MetricLogger(delimiter="  ")
    mlog.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    net.train()
    net = net.to(device)
    iters = 0
    if lr_decay_mode == 'iter':
      while True:
        mlog.clear()
        for inputs, labels in mlog.log_every(dataloader, display, 'train'):
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          acc1, = utils.accuracy(outputs, labels, topk=(1,))
          batch_size = inputs.shape[0]
          mlog.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
          mlog.meters['acc1'].update(acc1.item(), n=batch_size)
          iters += 1

          if iters % snapshot == 0:
              path = os.path.join(cfgs.get('train', 'snapshot_prefix'),
                                  cfgs.get('model', 'net'))
              utils.mkdir(path)
              torch.save(net.to('cpu').state_dict(), path + '/' + 'iter_%d.pt'%iters)
    elif lr_decay_mode == 'epoch':
        pass
    else:
        logging.fatal("Please specific lr_decay_mode. " +
                      "Currently only iter and epoch are accepted!")

    logging.info('Finished Training')

