from torchvision_sunner.utils import INFO
from torchvision_sunner.constant import *
import numpy as np
import torch

"""
    This script define some operation which are rather simple
    The operation only need to call function once (without inherit OP class)

    Author: SunnerLi
"""

class ToTensor():
    def __init__(self):
        """
            Change the tensor into torch.Tensor type
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor or other type. The tensor you want to deal with
        """
        if type(tensor) == np.ndarray:
            tensor = torch.from_numpy(tensor)
        return tensor

class ToFloat():
    def __init__(self):
        """
            Change the tensor into torch.FloatTensor
        """        
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        return tensor.float()

class Transpose():
    def __init__(self, direction = BHWC2BCHW):
        """
            Transfer the rank of tensor into target one

            Arg:    direction   - The direction you want to do the transpose
        """        
        self.direction = direction
        if self.direction == BHWC2BCHW:
            INFO("Applied << %15s >>, The rank format is BCHW" % self.__class__.__name__)
        elif self.direction == BCHW2BHWC:
            INFO("Applied << %15s >>, The rank format is BHWC" % self.__class__.__name__)
        else:
            raise Exception("Unknown direction symbol: {}".format(self.direction))

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if self.direction == BHWC2BCHW:
            tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        return tensor