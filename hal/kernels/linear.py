# linear.py

import math
import torch

__all__ = ['Linear', 'RFFLinear']


class Linear:
    def __init__(self):
        pass

    def __call__(self, x, y=None):
        if y is None:
            kernel = torch.mm(x, x.t())
        else:
            kernel = torch.mm(y, y.t())
        
        return kernel

class RFFLinear:
    def __init__(self):
        self.w_x = None
        self.b_x = None
        pass

    def __call__(self, x):
        return x
