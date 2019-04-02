from collections import Counter
from PIL import Image
from glob import glob
import numpy as np
import os

"""
    This script defines the function to read the containing of folder and read the file
    You should customize if your data is not considered in the torchvision_sunner previously

    Author: SunnerLi
"""

def readContain(folder_name):
    """
        Read the containing in the particular folder

        ==================================================================
        You should customize this function if your data is not considered
        ==================================================================

        Arg:    folder_name - The path of folder
        Ret:    The list of containing
    """
    # Check the common type in the folder
    common_type = Counter()
    for name in os.listdir(folder_name):
        common_type[name.split('.')[-1]] += 1
    common_type = common_type.most_common()[0][0]

    # Deal with the type
    if common_type == 'jpg':
        name_list = glob(os.path.join(folder_name, '*.jpg'))
    elif common_type == 'png':
        name_list = glob(os.path.join(folder_name, '*.png'))
    elif common_type == 'mp4':
        name_list = glob(os.path.join(folder_name, '*.mp4'))
    else:
        raise Exception("Unknown type {}, You should customize in read.py".format(common_type))
    return name_list

def readItem(item_name):
    """
        Read the file for the given item name

        ==================================================================
        You should customize this function if your data is not considered
        ==================================================================

        Arg:    item_name   - The path of the file
        Ret:    The item you read
    """
    file_type = item_name.split('.')[-1]
    if file_type == "png" or file_type == 'jpg':
        file_obj = np.asarray(Image.open(item_name))
        
        if len(file_obj.shape) == 3:
            # Ignore the 4th dim (RGB only)
            file_obj = file_obj[:, :, :3]
        elif len(file_obj.shape) == 2:
            # Make the rank of gray-scale image as 3
            file_obj = np.expand_dims(file_obj, axis = -1)
    return file_obj