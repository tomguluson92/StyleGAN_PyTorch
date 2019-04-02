from torchvision_sunner.data.base_dataset import BaseDataset
from torchvision_sunner.read import readContain, readItem
from torchvision_sunner.constant import *
from torchvision_sunner.utils import INFO

import torch.utils.data as Data

from PIL import Image
from glob import glob
import numpy as np
import subprocess
import random
import pickle
import torch
import math
import os

"""
    This script define the structure of video dataset

    =======================================================================================
    In the new version, we accept the form that the combination of video and folder:
    e.g. [[video1.mp4, image_folder]]
    On the other hand, the root can only be 'the list of list'
    You should use double list to represent different image domain.
    For example:
        [[video1.mp4], [video2.mp4]]                                => valid
        [[video1.mp4], [video_folder]]                              => valid
        [[video1.mp4, video2.mp4], [video_folder1, video_folder2]]  => valid
        [video1.mp4, video2.mp4]                                    => invalid!
    Also, the triple of nested list is not allow
    =======================================================================================

    Author: SunnerLi
"""

class VideoDataset(BaseDataset):
    def __init__(self, root = None, file_name = '.remain.pkl', T = 10, sample_method = UNDER_SAMPLING, transform = None, 
                    split_ratio = 0.0, decode_root = './.decode', save_file = False):
        """
            The constructor of VideoDataset

            Arg:    root            - The list object. The image set
                    file_name       - Str. The name of record file.
                    T               - Int. The maximun length of small video sequence
                    sample_method   - sunnerData.UNDER_SAMPLING or sunnerData.OVER_SAMPLING. Use down sampling or over sampling to deal with data unbalance problem.
                                      (default is sunnerData.OVER_SAMPLING)
                    transform       - transform.Compose object. You can declare some pre-process toward the image
                    split_ratio     - Float. The proportion to split the data. Usually used to split the testing data
                    decode_root     - Str. The path to store the ffmpeg decode result. 
                    save_file       - Bool. If storing the record file or not. Default is False
        """
        super().__init__()

        # Record the parameter
        self.root = root
        self.file_name = file_name
        self.T = T
        self.sample_method = sample_method
        self.transform = transform
        self.split_ratio = split_ratio
        self.decode_root = decode_root
        self.video_num = -1
        self.split_root  = None
        INFO()

        # Substitude the contain of record file if the record file is exist
        if not os.path.exists(file_name) and root is None:
            raise Exception("Record file {} not found. You should assign 'root' parameter!".format(file_name))
        elif os.path.exists(file_name):
            INFO("Load from file: {}".format(file_name))
            self.loadFromFile(file_name)      

        # Extend the images of folder into domain list
        self.extendFolder()

        # Split the image
        self.split()

        # Form the files obj
        self.getFiles()

        # Adjust the image number
        self.getVideoNum()

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
        return super().loadFromFile(file_name, 'video')

    def save(self, split_file_name = ".split.pkl"):
        """
            Save the information into record file

            Arg:    split_file_name - The path of record file which store the information of split data
        """
        super().save(self.file_name, self.split_ratio, split_file_name, 'video')

    # ===========================================================================================
    #       Define main function
    # ===========================================================================================
    def to_folder(self, name):
        """
            Transfer the name into the folder format
            e.g. 
                '/home/Dataset/video1_folder' => 'home_Dataset_video1_folder'
                '/home/Dataset/video1.mp4'    => 'home_Dataset_video1'

            Arg:    name    - Str. The path of file or original folder
            Ret:    The new (encoded) folder name
        """
        if not os.path.isdir(name):
            name = '_'.join(name.split('.')[:-1]) 
        domain_list = name.split('/')
        while True:
            if '.' in domain_list:
                domain_list.remove('.')
            elif '..' in domain_list:
                domain_list.remove('..')
            else:
                break
        return '_'.join(domain_list)

    def extendFolder(self):
        """
            Extend the video folder in root obj
        """
        if not self.files:
            # Extend the folder of video and replace as new root obj
            extend_root = []
            for domain in self.root:
                videos = []
                for video in domain:
                    if os.path.exists(video):
                        if os.path.isdir(video):
                            videos += readContain(video)
                        else:
                            videos.append(video)
                    else:
                        raise Exception("The path {} is not exist".format(videos))
                extend_root.append(videos)
            self.root = extend_root

    def split(self):
        """
            Split the root object into split_root object
            The original root object will shrink

            We also consider the case of pair image
            Thus we will check if the number of image in each domain is the same
            If it does, then we only generate the list once
        """
        # Check if the number of video in different domain is the same
        pairImage = True
        for domain_idx in range(len(self.root) - 1):
            if len(self.root[domain_idx]) != len(self.root[domain_idx + 1]):
                pairImage = False

        # Split the files
        self.split_root = []
        if pairImage:
            split_img_num = math.floor(len(self.root[0]) * self.split_ratio)
            choice_index_list = self.generateIndexList(range(len(self.root[0])), size = split_img_num)
        for domain_idx in range(len(self.root)):
            # determine the index list
            if not pairImage:
                split_img_num = math.floor(len(self.root[domain_idx]) * self.split_ratio)
                choice_index_list = self.generateIndexList(range(len(self.root[domain_idx])), size = split_img_num)
            # remove the corresponding term and add into new list
            split_img_list = []
            remain_img_list = self.root[domain_idx].copy()
            for j in choice_index_list:
                split_img_list.append(self.root[domain_idx][j])
            for j in choice_index_list:
                self.root[domain_idx].remove(remain_img_list[j])
            self.split_root.append(sorted(split_img_list))

    def getFiles(self):
        """
            Construct the files object for the assigned root
            We accept the user to mix folder with image
            This function can extract whole image in the folder
            
            However, unlike the setting in ImageDataset, we store the video result in root obj.
            Also, the 'images' name will be store in files obj

            The following list the progress of this function:
                1. check if we need to decode again
                2. decode if needed
                3. form the files obj
        """
        if not self.files:
            # Check if the decode process should be conducted again
            should_decode = not os.path.exists(self.decode_root)
            if not should_decode:
                for domain_idx, domain in enumerate(self.root):
                    for video in domain:
                        if not os.path.exists(os.path.join(self.decode_root, str(domain_idx), self.to_folder(video))):
                            should_decode = True
                            break

            # Decode the video if needed
            if should_decode:
                INFO("Decode from scratch...")
                if os.path.exists(self.decode_root):
                    subprocess.call(['rm', '-rf', self.decode_root])
                os.mkdir(self.decode_root)
                self.decodeVideo()
            else:
                INFO("Skip the decode process!")                

            # Form the files object
            self.files = {}
            for domain_idx, domain in enumerate(os.listdir(self.decode_root)):
                self.files[domain_idx] = []
                for video in os.listdir(os.path.join(self.decode_root, domain)):
                    self.files[domain_idx] += [
                        sorted(glob(os.path.join(self.decode_root, domain, video, "*")))
                    ]

    def decodeVideo(self):
        """
            Decode the single video into a series of images, and store into particular folder
        """
        for domain_idx, domain in enumerate(self.root):
            decode_domain_folder = os.path.join(self.decode_root, str(domain_idx))
            os.mkdir(decode_domain_folder)
            for video in domain:
                os.mkdir(os.path.join(self.decode_root, str(domain_idx), self.to_folder(video)))
                source = os.path.join(domain, video)
                target = os.path.join(decode_domain_folder, self.to_folder(video), "%5d.png")
                subprocess.call(['ffmpeg', '-i', source, target])

    def getVideoNum(self):
        """
            Obtain the video number in the loader for the specific sample method
            The function will check if the folder has been extracted
        """
        if self.video_num == -1:
            # Check if the folder has been extracted
            for domain in self.root:
                for video in domain:
                    if os.path.isdir(video):
                        raise Exception("You should extend the image in the folder {} first!" % video)

            # Statistic the image number
            for i, domain in enumerate(self.root):
                if i == 0:
                    self.video_num = len(domain)
                else:
                    if self.sample_method == OVER_SAMPLING:
                        self.video_num = max(self.video_num, len(domain))
                    elif self.sample_method == UNDER_SAMPLING:
                        self.video_num = min(self.video_num, len(domain))
        return self.video_num

    def print(self):
        """
            Print the information for each image domain
        """
        INFO()
        for domain in range(len(self.root)):
            total_frame = 0
            for video in self.files[domain]:
                total_frame += len(video)
            INFO("domain index: %d \tvideo number: %d\tframe total: %d" % (domain, len(self.root[domain]), total_frame))
        INFO()

    def __len__(self):
        return self.video_num

    def __getitem__(self, index):
        """
            Return single batch of data, and the rank is BTCHW
        """
        result = []
        for domain_idx in range(len(self.root)):

            # Form the sequence in single domain
            film_sequence = []
            max_init_frame_idx = len(self.files[domain_idx][index]) - self.T
            start_pos = random.randint(0, max_init_frame_idx)
            for i in range(self.T):
                img_path = self.files[domain_idx][index][start_pos + i]
                img = readItem(img_path)
                film_sequence.append(img)

            # Transform the film sequence
            film_sequence = np.asarray(film_sequence)
            if self.transform:
                film_sequence = self.transform(film_sequence)
            result.append(film_sequence)
        return result