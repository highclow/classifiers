import torch.utils.data 

from PIL import Image
import os



def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


class ImageList(torch.utils.data.Dataset):
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
