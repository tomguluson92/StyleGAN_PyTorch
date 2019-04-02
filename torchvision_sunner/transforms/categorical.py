from torchvision_sunner.utils import INFO
from torchvision_sunner.constant import *
from torchvision_sunner.transforms.simple import Transpose

from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import pickle
import torch
import json
import os

"""
    This script define the categorical-related operations, including:
        1. getCategoricalMapping
        2. CategoricalTranspose

    Author: SunnerLi
"""

# ----------------------------------------------------------------------------------------
#   Define the IO function toward pallete file
# ----------------------------------------------------------------------------------------

def load_pallete(file_name):
    """
        Load the pallete object from file

        Arg:    file_name   - The name of pallete .json file
        Ret:    The list of pallete object
    """
    # Load the list of dict from files (key is str)
    palletes_str_key = None
    with open(file_name, 'r') as f:
        palletes_str_key = json.load(f)

    # Change the key into color tuple
    palletes = [OrderedDict()] * len(palletes_str_key)
    for folder in range(len(palletes_str_key)):
        for key in palletes_str_key[folder].keys():
            tuple_key = list()
            for v in key.split('_'):
                tuple_key.append(int(v))
            palletes[folder][tuple(tuple_key)] = palletes_str_key[folder][key]
    return palletes

def save_pallete(pallete, file_name):
    """
        Load the pallete object from file

        Arg:    pallete     - The list of OrderDict objects
                file_name   - The name of pallete .json file
    """
    # Change the key into str
    pallete_str_key = [dict()] * len(pallete)
    for folder in range(len(pallete)):
        for key in pallete[folder].keys():

            str_key = '_'.join([str(_) for _ in key])
            pallete_str_key[folder][str_key] = pallete[folder][key]

    # Save into file
    with open(file_name, 'w') as f:
        json.dump(pallete_str_key, f)

# ----------------------------------------------------------------------------------------
#   Define the categorical-related operations
# ----------------------------------------------------------------------------------------

def getCategoricalMapping(loader = None, path = 'torchvision_sunner_categories_pallete.json'):
    """
        This function can statistic the different category with color
        And return the list of the mapping OrderedDict object

        Arg:    loader  - The ImageLoader object
                path    - The path of pallete file
        Ret:    The list of OrderDict object (palletes object)
    """
    INFO("Applied << %15s >>" % getCategoricalMapping.__name__)
    INFO("* Notice: the rank format of input tensor should be 'BHWC'")
    INFO("* Notice: The range of tensor should be in [0, 255]")
    if os.path.exists(path):
        palletes = load_pallete(path)
    else:
        INFO(">> Load from scratch, please wait...")

        # Get the number of folder
        folder_num = 0
        for img_list in loader:
            folder_num = len(img_list)
            break

        # Initialize the pallete list
        palletes = [OrderedDict()] * folder_num
        color_sets = [set()] * folder_num

        # Work
        for img_list in tqdm(loader):
            for folder_idx in range(folder_num):
                img = img_list[folder_idx]
                if torch.max(img) > 255 or torch.min(img) < 0:
                    raise Exception('tensor value out of range...\t range is [' + str(torch.min(img)) + ' ~ ' + str(torch.max(img)))
                img = img.cpu().data.numpy().astype(np.uint8)
                img = np.reshape(img, [-1, 3])
                color_sets[folder_idx] |= set([tuple(_) for _ in img])

        # Merge the color
        for i in range(folder_num):
            for color in color_sets[i]:
                if color not in palletes[i].keys():
                    palletes[i][color] = len(palletes[i])
        save_pallete(palletes, path)

    return palletes

class CategoricalTranspose():
    def __init__(self, pallete = None, direction = COLOR2INDEX, index_default = 0):
        """
            Transform the tensor into the particular format
            We support for 3 different kinds of format:
                1. one hot image
                2. index image
                3. color
            
            Arg:    pallete         - The pallete object (default is None)
                    direction       - The direction you want to change
                    index_default   - The default index if the color cannot be found in the pallete
        """
        self.pallete = pallete
        self.direction = direction
        self.index_default = index_default
        INFO("Applied << %15s >> , direction: %s" % (self.__class__.__name__, self.direction))
        INFO("* Notice: The range of tensor should be in [-1, 1]")
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def fn_color_to_index(self, tensor):
        """
            Transfer the tensor from the RGB colorful format into the index format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with index format
        """
        if self.pallete is None:
            raise Exception("The direction << %s >> need the pallete object" % self.direction)
        tensor = tensor.transpose(-3, -2).transpose(-2, -1).cpu().data.numpy()
        size_tuple = list(np.shape(tensor))
        tensor = (tensor * 127.5 + 127.5).astype(np.uint8)
        tensor = np.reshape(tensor, [-1, 3])
        tensor = [tuple(_) for _ in tensor]
        tensor = [self.pallete.get(_, self.index_default) for _ in tensor]
        tensor = np.asarray(tensor)
        size_tuple[-1] = 1
        tensor = np.reshape(tensor, size_tuple)
        tensor = torch.from_numpy(tensor).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def fn_index_to_one_hot(self, tensor):
        """
            Transfer the tensor from the index format into the one-hot format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with one-hot format
        """
        # Get the number of classes
        tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        size_tuple = list(np.shape(tensor))
        tensor = tensor.view(-1).cpu().data.numpy()
        channel = np.amax(tensor) + 1

        # Get the total number of pixel
        num_of_pixel = 1
        for i in range(len(size_tuple) - 1):
            num_of_pixel *= size_tuple[i]

        # Transfer as ont-hot format
        one_hot_tensor = np.zeros([num_of_pixel, channel])
        for i in range(channel):
            one_hot_tensor[tensor == i, i] = 1

        # Recover to origin rank format and shape
        size_tuple[-1] = channel
        tensor = np.reshape(one_hot_tensor, size_tuple)
        tensor = torch.from_numpy(tensor).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def fn_one_hot_to_index(self, tensor):
        """
            Transfer the tensor from the one-hot format into the index format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with index format
        """
        _, tensor = torch.max(tensor, dim = 1)
        tensor = tensor.unsqueeze(1)
        return tensor

    def fn_index_to_color(self, tensor):
        """
            Transfer the tensor from the index format into the RGB colorful format

            Arg:    tensor  - The tensor obj. The tensor you want to deal with
            Ret:    The tensor with RGB colorful format
        """
        if self.pallete is None:
            raise Exception("The direction << %s >> need the pallete object" % self.direction)
        tensor = tensor.transpose(-3, -2).transpose(-2, -1).cpu().data.numpy()
        reverse_pallete = {self.pallete[x]: x for x in self.pallete}
        batch, height, width, channel = np.shape(tensor)
        tensor = np.reshape(tensor, [-1])
        tensor = np.round(tensor, decimals=0)
        tensor = np.vectorize(reverse_pallete.get)(tensor)
        tensor = np.reshape(np.asarray(tensor).T, [batch, height, width, len(reverse_pallete[0])])
        tensor = torch.from_numpy((tensor - 127.5) / 127.5).transpose(-1, -2).transpose(-2, -3)
        return tensor

    def __call__(self, tensor):
        if self.direction == COLOR2INDEX:
            return self.fn_color_to_index(tensor)
        elif self.direction == INDEX2COLOR:
            return self.fn_index_to_color(tensor)
        elif self.direction == ONEHOT2INDEX:
            return self.fn_one_hot_to_index(tensor)
        elif self.direction == INDEX2ONEHOT:
            return self.fn_index_to_one_hot(tensor)
        elif self.direction == ONEHOT2COLOR:
            return self.fn_index_to_color(self.fn_one_hot_to_index(tensor))
        elif self.direction == COLOR2ONEHOT:
            return self.fn_index_to_one_hot(self.fn_color_to_index(tensor))
        else:
            raise Exception("Unknown direction: {}".format(self.direction))