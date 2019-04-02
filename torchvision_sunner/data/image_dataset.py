from torchvision_sunner.data.base_dataset import BaseDataset
from torchvision_sunner.read import readContain, readItem
from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO

from skimage import io as io
# from PIL import Image
from glob import glob

import torch.utils.data as Data

import pickle
import math
import os

"""
    This script define the structure of image dataset

    =======================================================================================
    In the new version, we accept the form that the combination of image and folder:
    e.g. [[image1.jpg, image_folder]]
    On the other hand, the root can only be 'the list of list'
    You should use double list to represent different image domain.
    For example:
        [[image1.jpg], [image2.jpg]]                                => valid
        [[image1.jpg], [image_folder]]                              => valid
        [[image1.jpg, image2.jpg], [image_folder1, image_folder2]]  => valid
        [image1.jpg, image2.jpg]                                    => invalid!
    Also, the triple of nested list is not allow
    =======================================================================================

    Author: SunnerLi
"""

class ImageDataset(BaseDataset):
    def __init__(self, root = None, file_name = '.remain.pkl', sample_method = UNDER_SAMPLING, transform = None, 
                    split_ratio = 0.0, save_file = False):
        """
            The constructor of ImageDataset

            Arg:    root            - The list object. The image set
                    file_name       - The str. The name of record file. 
                    sample_method   - sunnerData.UNDER_SAMPLING or sunnerData.OVER_SAMPLING. Use down sampling or over sampling to deal with data unbalance problem.
                                      (default is sunnerData.OVER_SAMPLING)
                    transform       - transform.Compose object. You can declare some pre-process toward the image
                    split_ratio     - Float. The proportion to split the data. Usually used to split the testing data
                    save_file       - Bool. If storing the record file or not. Default is False
        """
        super().__init__()
        # Record the parameter
        self.root = root
        self.file_name = file_name
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        self.save_file = save_file
        self.img_num = -1
        INFO()

        # Substitude the contain of record file if the record file is exist
        if os.path.exists(file_name) and self.loadFromFile(file_name):
            self.getImgNum()
        elif not os.path.exists(file_name) and root is None:
            raise Exception("Record file {} not found. You should assign 'root' parameter!".format(file_name))
        else:   
            # Extend the images of folder into domain list
            self.getFiles()

            # Change root obj as the index format
            self.root = range(len(self.root))

            # Adjust the image number
            self.getImgNum()

            # Split the files if split_ratio is more than 0.0
            self.split()       

            # Save the split information
            self.save() 

        # Print the domain information
        self.print()

    # ===========================================================================================
    #       Define IO function
    # ===========================================================================================
    def loadFromFile(self, file_name):
        """
            Load the root and files information from .pkl record file
            This function will return False if the record file format is invalid

            Arg:    file_name   - The name of record file
            Ret:    If the loading procedure are successful or not
        """
        return super().loadFromFile(file_name, 'image')

    def save(self, split_file_name = ".split.pkl"):
        """
            Save the information into record file

            Arg:    split_file_name - The path of record file which store the information of split data
        """
        super().save(self.file_name, self.split_ratio, split_file_name, 'image')

    # ===========================================================================================
    #       Define main function
    # ===========================================================================================
    def getFiles(self):
        """
            Construct the files object for the assigned root
            We accept the user to mix folder with image
            This function can extract whole image in the folder
            The element in the files will all become image 

            *******************************************************
            * This function only work if the files object is None *
            *******************************************************
        """
        if not self.files:
            self.files = {}
            for domain_idx, domain in enumerate(self.root):
                images = []
                for img in domain:
                    if os.path.exists(img):
                        if os.path.isdir(img):
                            images += readContain(img)
                        else:
                            images.append(img)
                    else:
                        raise Exception("The path {} is not exist".format(img))
                self.files[domain_idx] = sorted(images)

    def getImgNum(self):
        """
            Obtain the image number in the loader for the specific sample method
            The function will check if the folder has been extracted
        """
        if self.img_num == -1:
            # Check if the folder has been extracted
            for domain in self.root:
                for img in self.files[domain]:
                    if os.path.isdir(img):
                        raise Exception("You should extend the image in the folder {} first!" % img)

            # Statistic the image number
            for domain in self.root:
                if domain == 0:
                    self.img_num = len(self.files[domain])
                else:
                    if self.sample_method == OVER_SAMPLING:
                        self.img_num = max(self.img_num, len(self.files[domain]))
                    elif self.sample_method == UNDER_SAMPLING:
                        self.img_num = min(self.img_num, len(self.files[domain]))
        return self.img_num

    def split(self):
        """
            Split the files object into split_files object
            The original files object will shrink

            We also consider the case of pair image
            Thus we will check if the number of image in each domain is the same
            If it does, then we only generate the list once
        """
        # Check if the number of image in different domain is the same
        if not self.files:
            self.getFiles()
        pairImage = True
        for domain in range(len(self.root) - 1):
            if len(self.files[domain]) != len(self.files[domain + 1]):
                pairImage = False

        # Split the files
        self.split_files = {}
        if pairImage:
            split_img_num = math.floor(len(self.files[0]) * self.split_ratio)
            choice_index_list = self.generateIndexList(range(len(self.files[0])), size = split_img_num)
        for domain in range(len(self.root)):
            # determine the index list
            if not pairImage:
                split_img_num = math.floor(len(self.files[domain]) * self.split_ratio)
                choice_index_list = self.generateIndexList(range(len(self.files[domain])), size = split_img_num)
            # remove the corresponding term and add into new list
            split_img_list = []
            remain_img_list = self.files[domain].copy()
            for j in choice_index_list:
                split_img_list.append(self.files[domain][j])
            for j in choice_index_list:
                self.files[domain].remove(remain_img_list[j])
            self.split_files[domain] = sorted(split_img_list)

    def print(self):
        """
            Print the information for each image domain
        """
        INFO()
        for domain in range(len(self.root)):
            INFO("domain index: %d \timage number: %d" % (domain, len(self.files[domain])))
        INFO()

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        return_list = []
        for domain in self.root:
            img_path = self.files[domain][index]
            img = readItem(img_path)
            if self.transform:
                img = self.transform(img)
            return_list.append(img)
        return return_list