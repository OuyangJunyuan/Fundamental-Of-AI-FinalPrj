import os
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms as T ,datasets


class MyDataset(object):  # (data.Dataset):

    def __init__(self, train=True, test=False):#,torch_dataset = None , root = None, transforms=None, train=True, test=True):
        # 目前是使用 torch 自带的数据集，后续更改本类为使用自定义数据集

        self.test = test

        self.transforms = T.Compose([T.ToTensor(),  # PIL转换为pytorch.tensor 且会把(0,255)-> (0,1)
                                    T.Normalize((0.5,), (0.5,))]  # 标准化至(-1,1)
                                    )  # 使用类似t.nn.sequential一样的操作来将一系列TF组合成为1个大TF通过Compose
        if self.test:
            self.__testSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False,  # 获取数据集
                                                 transform=self.transforms)
            self.testSetLoader = data.DataLoader(self.__testSet, batch_size=64, shuffle=True)   # 加载数据集，使用DataLoader
        elif train:
            self.__trainSet = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True,
                                                  transform=self.transforms)
            self.trainSetLoader = data.DataLoader(self.__trainSet, batch_size=64, shuffle=True)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


