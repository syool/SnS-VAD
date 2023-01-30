import torch.nn as nn
import torch.nn.init as init


class Initialize(nn.Module):
    init = {
            'normal': init.normal_,
            'uniform': init.uniform_,
            'kaiming': init.kaiming_normal_,
            'kaiming_uniform': init.kaiming_uniform_,
            'xavier': init.xavier_normal_,
            'xavier_uniform': init.xavier_uniform_
        }
    
    def __init__(self, method) -> None:
        super(Initialize, self).__init__()
        self.init_method = self.init[method]
    
    def weights_init(self, m):
        layer = m.__class__.__name__
        if layer.find('Conv') != -1:
            self.init_method(m.weight.data)
        elif layer.find('BatchNorm') != -1:
            self.init_method(m.weight.data)
            init.zeros_(m.bias.data)
    
    def __call__(self, m):
        self.weights_init(m)