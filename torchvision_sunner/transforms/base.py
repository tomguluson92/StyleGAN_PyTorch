import numpy as np
import torch

"""
    This class define the parent class of operation

    Author: SunnerLi
"""

class OP():
    """
        The parent class of each operation
        The goal of this class is to adapting with different input format
    """
    def work(self, tensor):
        """
            The virtual function to define the process in child class

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
        """
        raise NotImplementedError("You should define your own function in the class!")

    def __call__(self, tensor):
        """
            This function define the proceeding of the operation
            There are different choice toward the tensor parameter
            1. torch.Tensor and rank is CHW
            2. np.ndarray and rank is CHW
            3. torch.Tensor and rank is TCHW
            4. np.ndarray and rank is TCHW

            Arg:    tensor  - The tensor you want to operate
            Ret:    The operated tensor
        """
        isTensor = type(tensor) == torch.Tensor
        if isTensor:
            tensor_type = tensor.type()
            tensor = tensor.cpu().data.numpy()
        if len(tensor.shape) == 3:
            tensor = self.work(tensor)
        elif len(tensor.shape) == 4:
            tensor = np.asarray([self.work(_) for _ in tensor])
        else:
            raise Exception("We dont support the rank format {}".format(tensor.shape),
                "If the rank of the tensor shape is only 2, you can call 'GrayStack()'")
        if isTensor:
            tensor = torch.from_numpy(tensor)
            tensor = tensor.type(tensor_type)
        return tensor