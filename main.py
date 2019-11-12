import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(module)s: %(message)s',
                    level=logging.DEBUG)

import utils
from configs import get_config
from datasets import ImageList
from models import resnet50


def get_command():
    parser = argparse.ArgumentParser(description=
                                     'Process for training classifier')
    parser.add_argument('-c','--config', type=str, required=True,
                        help='configure file')
    args = parser.parse_args()
    return args


def get_dataloader(cfgs, split):
    transform = transforms.Compose(
            [transforms.ToTensor(),
            ])

    dataset = ImageList(root=cfgs.get(split,'root'),
                        imagelist=cfgs.get(split,'imagelist'),
                        transform = transform)
    loader = DataLoader(dataset,
                        batch_size=cfgs.getint(split, 'batch_size'),
                        shuffle=cfgs.getboolean(split, 'shuffle'),
                        num_workers=cfgs.getint(split, 'num_workers'))
    return loader

def get_net(net_name, param_path):
    if net_name == 'ResNet50':
        return resnet50(pretrained=False, num_classes=3)

def get_optimizer(net, cfgs):
    if cfgs.get('train', 'optimizer') == 'SGD':
        lr = cfgs.getfloat('train', 'base_lr')
        momentum = cfgs.getfloat('train', 'momentum')
        weight_decay = cfgs.getfloat('train', 'weight_decay')
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    return optimizer


def train(net, criterion, dataloader, optimizer, args):
    lr_decay_mode = args.get('train', 'lr_decay_mode')
    device = args.get('model', 'device')
    display = args.getint('train', 'display')
    mlog = utils.MetricLogger(delimiter="  ")
    mlog.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    if lr_decay_mode == 'iter':
      while True:
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          acc1 = utils.accuracy(outputs, labels, topk=(1,))
          # print statistics
          mlog.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
          mlog.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
          if i % display == display - 1:
              print('[, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / display))
              running_loss = 0.0
    elif lr_decay_mode == 'epoch':
        pass
    else:
        logging.fatal("Please specific lr_decay_mode. " +
                      "Currently only iter and epoch are accepted!")
    
    print('Finished Training')


def main(cfgs):

    trainloader = get_dataloader(cfgs, 'train')
    testloader = get_dataloader(cfgs, 'test')

    net = get_net(cfgs.get('model', 'net'), cfgs.get('model', 'params'))
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(net, cfgs)

    train(net, criterion, trainloader, optimizer, cfgs)

if __name__ == '__main__':
    args = get_command()
    logging.debug(args)
    
    cfgs = get_config(args.config)
    main(cfgs)
    

    
