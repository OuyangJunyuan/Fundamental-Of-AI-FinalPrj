import torch as t
import time
import os


class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()     # 初始化父类
        self.model_name = 'model'   #
        print(self.model_name)

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            abspath = os.getcwd()
            prefix = abspath + '/data/checkpoint/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        else:
            abspath = os.getcwd()
            prefix = abspath + '/data/checkpoint/' + name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
