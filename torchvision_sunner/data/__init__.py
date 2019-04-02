"""
    This script define the wrapper of the Torchvision_sunner.data

    Author: SunnerLi
"""
from torch.utils.data import DataLoader

from torchvision_sunner.data.image_dataset import ImageDataset
from torchvision_sunner.data.video_dataset import VideoDataset
from torchvision_sunner.data.loader import *
from torchvision_sunner.constant import *
from torchvision_sunner.utils import *