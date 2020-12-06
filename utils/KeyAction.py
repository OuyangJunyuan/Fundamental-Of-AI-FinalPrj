import myNet
from data.dataset import MyDataset
from config import DefaultConfig as Cfg

import torch
from torch import optim
from torchvision import models
option = Cfg()


def train(**kwargs):
    # 定义训练过程
    option.update_cfg(**kwargs)
    print('--------------')
    print('Running func: train')
    # 模型：
    # net_type = getattr(myNet, option.model)  # 通过属性名字符串从一个对象中获取一个属性对象, hasattr() 判断是否有这个成员,setattr()设置对象的某个的属性对象

    model = myNet.BriefNet()  # net_type()  # 获取的是类对象，后面需要加上()变成构造函数
    if option.use_gpu:
        model.cuda()

    # 数据:
    train_dataset = MyDataset(train=True)
    val_dataset = MyDataset(test=True)

    # 目标函数和优化器
    if option.use_gpu:
        criterion = torch.nn.NLLLoss().cuda()
    else:
        criterion = torch.nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=option.lr, momentum=0.8)  # , weight_decay=option.lr_decay) # 慎用，
    # 在pytorch书上加上这个了，但是加上后可能学习率->0 导致loss不下降

    train_losses, test_losses, test_accuracy = [], [], []
    # 训练
    model.train()

    for epoch in range(option.max_epoch):
        running_loss = 0
        for ii, (image, label) in enumerate(train_dataset.trainSetLoader):
            optimizer.zero_grad()
            x = image
            target = label
            if option.use_gpu:
                x = x.cuda()
                target = target.cuda()

            ps = model(x)
            loss = criterion(ps, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            train_losses.append(running_loss / len(train_dataset.trainSetLoader))

            test_loss, accuracy = val(model, criterion, val_dataset.testSetLoader)
            test_losses.append(test_loss)
            test_accuracy.append(accuracy)

            print("Epoch: {}/{}.. ".format(epoch + 1, option.max_epoch),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(test_accuracy[-1]))
    model.save(option.save_as)
    return train_losses, test_losses, test_accuracy


def val(model, criterion, data_loader):
    # model = torch.nn.Module()  # 为了补全，代码写完记得注释掉

    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels in data_loader:
            val_input = images if not option.use_gpu else images.cuda()
            val_labels = labels if not option.use_gpu else labels.cuda()

            ps = model(val_input)
            test_loss += criterion(ps, val_labels)

            top_p, top_class = torch.exp(ps).topk(1, dim=1)
            equals = top_class == val_labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        if option.use_gpu:
            model.cuda()
            criterion.cuda()

    # 训练结束保存网络参数，训练太久可能发生过拟合。可以在大约8-10个训练时期停止， 此策略称为提前停止。
    # 实际上，您在训练时会经常保存模型，然后选择验证损失最小的模型。
    model.train()
    return test_loss.cpu()/len(data_loader), accuracy.cpu()/len(data_loader)

