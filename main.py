import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(module)s: %(message)s',
                    level=logging.INFO)

import utils
from configs import get_config
from datasets import ImageList
from models import resnet50


def get_command():
    parser = argparse.ArgumentParser(description=
                                     'Process for training classifier')
    parser.add_argument('-c','--config', type=str, required=True,
                        help='configure file')
    parser.add_argument('--test', action='store_true',
                        help='test the network')
    args = parser.parse_args()
    return args


def get_dataloader(cfgs, split):
    if split == 'train':
      transform = transforms.Compose(
              [transforms.Resize(256),
#               RandomRotation(15),
               transforms.RandomCrop(224),
               transforms.RandomHorizontalFlip(),
#               transforms.RandomErasing(),
               transforms.ToTensor(),
              ])
    else:
      transform = transforms.Compose(
              [transforms.Resize(256),
               transforms.CenterCrop(224),
               transforms.ToTensor(),
              ])

    dataset = ImageList(root=cfgs.get(split,'root'),
                        imagelist=cfgs.get(split,'imagelist'),
                        transform = transform)
    loader = DataLoader(dataset,
                        batch_size=cfgs.getint(split, 'batch_size'),
                        shuffle=cfgs.getboolean(split, 'shuffle'),
                        num_workers=cfgs.getint(split, 'num_workers'))
    return loader


def get_net(net_name, param_path, device='cpu'):
    if net_name == 'ResNet50':
      net = resnet50(num_classes=3)
      if param_path:
        params = torch.load(param_path, map_location=device)
        for n in net.state_dict().keys():
          if n in params and params[n].shape == net.state_dict()[n].shape:
            net.state_dict()[n][...] = params[n]
      return net


def get_optimizer(net, cfgs):
    if cfgs.get('train', 'optimizer') == 'SGD':
      lr = cfgs.getfloat('train', 'base_lr')
      momentum = cfgs.getfloat('train', 'momentum')
      weight_decay = cfgs.getfloat('train', 'weight_decay')
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
    return optimizer


def train(net, criterion, dataloader, optimizer, cfgs):
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
        running_loss = 0.0
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
    
    print('Finished Training')



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
        running_loss = 0.0
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

    print('Finished Training')



def train_net(cfgs):
    if cfgs.get('model', 'device') == 'cuda':
      device_id = cfgs.getint('model', 'device_id')
      torch.cuda.set_device(device_id)

    logging.info('Create Data Loader')
    trainloader = get_dataloader(cfgs, 'train')

    logging.info('Create Network')
    net = get_net(cfgs.get('model', 'net'), cfgs.get('model', 'params'))
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(net, cfgs)

    logging.info('Train Network')
    train(net, criterion, trainloader, optimizer, cfgs)


def test_net(cfgs):
    device = cfgs.get('model', 'device')
    if device == 'cuda':
      device_id = cfgs.getint('model', 'device_id')
      torch.cuda.set_device(device_id)

    logging.info('Create Data Loader')
    testloader = get_dataloader(cfgs, 'test')

    logging.info('Create Network')
    path = os.path.join(cfgs.get('train', 'snapshot_prefix'),
                        cfgs.get('model', 'net'))

    mlog = utils.MetricLogger(delimiter="  ")
    for d in os.listdir(path):
      if '12' in d or '10' in d:
        continue
      model_path = os.path.join(path, d)
      logging.info('Current Model: %s'%model_path)
      net = get_net(cfgs.get('model', 'net'), model_path).to(device)
      net.eval()

      for inputs, labels in mlog.log_every(testloader, 1, 'test'):
        inputs = inputs.to(device)
        labels = labels.to(device)
  
        # forward 
        outputs = net(inputs)
        outputs = F.softmax(outputs, dim=1)
        print(outputs, labels)
  
        # print statistics
        acc1, = utils.accuracy(outputs, labels, topk=(1,))
        batch_size = inputs.shape[0]
        mlog.meters['acc1'].update(acc1.item(), n=batch_size)
  

if __name__ == '__main__':
    args = get_command()
    logging.info(args)
    cfgs = get_config(args.config)
    if args.test:
      logging.info('Test')
      test_net(cfgs)
    else:
      logging.info('Train')
      train_net(cfgs)
    

    
