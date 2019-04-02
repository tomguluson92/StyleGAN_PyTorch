from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO
import torch.utils.data as Data

import pickle
import random
import os

"""
    This script define the parent class to deal with some common function for Dataset

    Author: SunnerLi
"""

class BaseDataset(Data.Dataset):
    def __init__(self):
        self.save_file = False
        self.files = None
        self.split_files = None

    def generateIndexList(self, a, size):
        """
            Generate the list of index which will be picked
            This function will be used as train-test-split

            Arg:    a       - The list of images
                    size    - Int, the length of list you want to create
            Ret:    The index list
        """
        result = set()
        while len(result) != size:
            result.add(random.randint(0, len(a) - 1))
        return list(result)

    def loadFromFile(self, file_name, check_type = 'image'):
        """
            Load the root and files information from .pkl record file
            This function will return False if the record file format is invalid

            Arg:    file_name   - The name of record file
                    check_type  - Str. The type of the record file you want to check
            Ret:    If the loading procedure are successful or not
        """
        with open(file_name, 'rb') as f:
            obj = pickle.load(f)
            self.type  = obj['type']
            if self.type == check_type:
                INFO("Load from file: {}".format(file_name))
                self.root  = obj['root']
                self.files = obj['files']
                return True
            else:
                INFO("Record file type: {}\tFail to load...".format(self.type))
                INFO("Form the contain from scratch...")
                return False

    def save(self, remain_file_name, split_ratio, split_file_name = ".split.pkl", save_type = 'image'):
        """
            Save the information into record file

            Arg:    remain_file_name    - The path of record file which store the information of remain data
                    split_ratio         - Float. The proportion to split the data. Usually used to split the testing data
                    split_file_name     - The path of record file which store the information of split data
                    save_type           - Str. The type of the record file you want to save
        """
        if self.save_file:
            if not os.path.exists(remain_file_name):
                with open(remain_file_name, 'wb') as f:
                    pickle.dump({
                        'type': save_type,
                        'root': self.root,
                        'files': self.files
                    }, f)
            if split_ratio:
                INFO("Split the dataset, and save as {}".format(split_file_name))
                with open(split_file_name, 'wb') as f:
                    pickle.dump({
                        'type': save_type,
                        'root': self.root,
                        'files': self.split_files
                    }, f) 