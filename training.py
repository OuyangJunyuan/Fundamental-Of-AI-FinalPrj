from utils.KeyAction import train
from matplotlib import pyplot as plt


train_losses, test_losses, test_accuracy = train(use_gpu=True, lr=0.003, max_epoch=40, save_as='SGD_DropOut')


plt.grid(ls="--")
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()



