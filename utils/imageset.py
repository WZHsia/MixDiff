import torch
import torchvision.transforms.functional as TF
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myResize:
    def __init__(self, size_h=224, size_w=224):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


def file_read(file):
    name = os.path.splitext(file)[1]
    if name == ".png" or name == ".jpg":
        rst = Image.open(file)
    elif name == ".npy":
        rst = np.load(file)
    return rst


class ImageLoader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.img_path = os.path.join(data_path, 'image')
        self.mask_path = os.path.join(os.path.join(data_path, 'mask'))
        self.imgs = os.listdir(self.img_path)
        self.masks = os.listdir(self.mask_path)
        self.transformer = transforms.Compose([
                            myToTensor(),
                            myRandomHorizontalFlip(p=0.5),
                            myRandomVerticalFlip(p=0.5),
                            myResize(256, 256)
                        ])

    def __getitem__(self, index):
        image_path = self.imgs[index]
        mask_path = self.masks[index]
        img = np.array(file_read(os.path.join(self.img_path, image_path))) / 255
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        if len(img[0][0]) > 3:
            img = img[:, :, :3]
        msk = np.array(file_read(os.path.join(self.mask_path, mask_path)).convert('L')) / 255
        msk = np.where(msk > 0.5, 1.0, 0)
        if msk.ndim == 2:
            msk = np.expand_dims(msk, axis=2)
        # msk = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs)


if __name__ == "__main__":
    isbi_dataset = ImageLoader("../data/BUSI")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
