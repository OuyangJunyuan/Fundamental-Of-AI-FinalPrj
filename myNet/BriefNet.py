import torch as t
import myNet.BasicModule as BM
from torch import nn, optim  # 导入 网络工具箱和优化器
from torch.nn import functional as F  # 导入 torch 自带的functional(计算图中的计算方块，定义了前向和后向传播函数)


# 自定义网络, 声明一堆网络层 然后在 forward中定义其连接方式
class BriefNet(BM.BasicModule):
    def __init__(self):
        super(BriefNet, self).__init__()
        # nn.ReLU 和 functional.relu的差别（大部分nn.layer 都对应一个nn.functional中的函数）：
        # 前者是nn.Module的子类，后者是nn.functional的子类
        # nn.中许多layer和激活函数都是继承与nn.Module的可以自动提取派生类中的可学习参数(通过继承Module实现的__getattr__和__setattr__)
        # functional是没有可学习参数的，可以用在激活函数、池化等地方
        self.fc = nn.Sequential(
            nn.Linear(784, 256),  # 数据集图片为28*28单通道，此处直接用多层感知机而不是用CNN
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 定义前向传播函数
        x = x.view(x.shape[0], -1)  # 将输入图片变成列向量输入
        x = self.fc(x)
        return x




