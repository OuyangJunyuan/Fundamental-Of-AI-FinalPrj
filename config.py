import warnings


class DefaultConfig(object):
    model = 'BriefNet'
    save_as = 'BriefNet'

    batch_size = 64
    use_gpu = True
    num_workers = 4
    result_file = 'result.csv'


    max_epoch = 10

    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    def update_cfg(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning： has not attribute %s" % k)
            setattr(self, k, v)

        print('user config: ')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):  # 如果不是私有属性
                print(k, getattr(self, k))
