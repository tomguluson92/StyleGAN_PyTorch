from torchvision_sunner.transforms.base import OP
from torchvision_sunner.utils import INFO
from skimage import transform
import numpy as np
import torch

"""
    This script define some complex operations
    These kind of operations should conduct work iteratively (with inherit OP class)

    Author: SunnerLi
"""

class Resize(OP):
    def __init__(self, output_size):
        """
            Resize the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
        """
        self.output_size = output_size
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BHWC'")

    def work(self, tensor):
        """
            Resize the tensor
            If the tensor is not in the range of [-1, 1], we will do the normalization automatically

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The resized tensor
        """
        # Normalize the tensor if needed
        mean, std = -1, -1
        min_v = np.min(tensor)
        max_v = np.max(tensor)
        if not (max_v <= 1 and min_v >= -1):
            mean = 0.5 * max_v + 0.5 * min_v
            std  = 0.5 * max_v - 0.5 * min_v
            # print(max_v, min_v, mean, std)
            tensor = (tensor - mean) / std

        # Work
        tensor = transform.resize(tensor, self.output_size, mode = 'constant', order = 0)

        # De-normalize the tensor
        if mean != -1 and std != -1:
            tensor = tensor * std + mean
        return tensor    

class Normalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Normalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the result will locate in [-1, 1]

            Args:
                mean        - The mean of the result tensor
                std         - The standard deviation
        """
        self.mean = mean
        self.std  = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        INFO("*****************************************************************")
        INFO("* Notice: You should must call 'ToFloat' before normalization")
        INFO("*****************************************************************")
        if self.mean == [127.5, 127.5, 127.5] and self.std == [127.5, 127.5, 127.5]:
            INFO("* Notice: The result will locate in [-1, 1]")

    def work(self, tensor):
        """
            Normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append((t - m) / s)
        tensor = np.asarray(result)

        # Check if the normalization can really work
        if np.min(tensor) < -1 or np.max(tensor) > 1:
            raise Exception("Normalize can only work with float tensor",
                "Try to call 'ToFloat()' before normalization")
        return tensor

class UnNormalize(OP):
    def __init__(self, mean = [127.5, 127.5, 127.5], std = [127.5, 127.5, 127.5]):
        """
            Unnormalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the function will assume that the original distribution locates in [-1, 1]

            Args:
                mean    - The mean of the result tensor
                std     - The standard deviation
        """
        self.mean = mean
        self.std = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if self.mean == [127.5, 127.5, 127.5] and self.std == [127.5, 127.5, 127.5]:
            INFO("* Notice: The function assume that the input range is [-1, 1]")

    def work(self, tensor):
        """
            Un-normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The un-normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append(t * s + m)
        tensor = np.asarray(result)
        return tensor

class ToGray(OP):
    def __init__(self):
        """
            Change the tensor as the gray scale
            The function will turn the BCHW tensor into B1HW gray-scaled tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def work(self, tensor):
        """
            Make the tensor into gray-scale

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The gray-scale tensor, and the rank of the tensor is B1HW
        """
        if tensor.shape[0] == 3:
            result = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            result = np.expand_dims(result, axis = 0)
        elif tensor.shape[0] != 4:
            result = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
            result = np.expand_dims(result, axis = 1)
        else:
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        return result