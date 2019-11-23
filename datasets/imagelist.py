import os
import logging
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


def default_transform():
    transform = transforms.Compose(
        [transforms.Resize(144),
         transforms.CenterCrop(128),
         transforms.ToTensor(),
        ])
    return transform


class ImageList(Dataset):
    def __init__(self, root, imagelist, transform=None):
        self.root      = root
        self.imgList   = default_list_reader(imagelist)
        self.transform = transform


    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img_loc=os.path.join(self.root, imgPath)
        img = Image.open(img_loc)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgList)


def get_imagelist_dataloader(cfgs, split):
    logging.info('Create %s data loader from %s'%(split,cfgs.get(split,'imagelist')))
    if split == 'train':
      transform = transforms.Compose(
              [transforms.Resize(144),
               #transforms.ColorJitter(brightness=10),
               #transforms.ColorJitter(brightness=10, contrast=10,
               #                       saturation=10, hue),
               transforms.RandomRotation(30),
               transforms.RandomCrop(128),
               transforms.RandomHorizontalFlip(),
#               transforms.RandomErasing(),
               transforms.ToTensor(),
              ])
    else:
      transform = default_transform()

    dataset = ImageList(root=cfgs.get(split,'root'),
                        imagelist=cfgs.get(split,'imagelist'),
                        transform = transform)
    loader = DataLoader(dataset,
                        batch_size=cfgs.getint(split, 'batch_size'),
                        shuffle=cfgs.getboolean(split, 'shuffle'),
                        num_workers=cfgs.getint(split, 'num_workers'))
    return loader
