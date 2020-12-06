import torch
import myNet
import data.dataset as data

import matplotlib.pyplot as plt
import numpy as np


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)


val_dataset = data.MyDataset(test=True)
model = myNet.BriefNet()
model.load('/home/ou/workspace/pycharm_ws/AI_FinalPrj/data/checkpoint/sgd+dropout_1206_00:18:48.pth')
model.eval()

dataiter = iter(val_dataset.testSetLoader)
images, labels = dataiter.next()
img = images[0]
img = img.view(1, 784)
with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)


view_classify(img.view(1, 28, 28), ps, version='Fashion')
plt.show()