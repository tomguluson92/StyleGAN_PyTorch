from torchvision_sunner.constant import *
from collections import Iterator
import torch.utils.data as data

"""
    This script define the extra loader, and it can be used in flexibility. The loaders include:
        1. ImageLoader (The old version exist)
        2. MultiLoader
        3. IterationLoader

    Author: SunnerLi
"""

class ImageLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers = 1):
        """
            The DataLoader object which can deal with ImageDataset object.

            Arg:    dataset     - ImageDataset. You should use sunnerData.ImageDataset to generate the instance first
                    batch_size  - Int.
                    shuffle     - Bool. Shuffle the data or not
                    num_workers - Int. How many thread you want to use to read the batch data
        """
        super(ImageLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers)
        self.dataset = dataset
        self.iter_num = self.__len__()

    def __len__(self):       
        return round(self.dataset.img_num / self.batch_size)

    def getImageNumber(self):
        return self.dataset.img_num

class MultiLoader(Iterator):
    def __init__(self, datasets, batch_size=1, shuffle=False, num_workers = 1):
        """
            This class can deal with multiple dataset object

            Arg:    datasets    - The list of ImageDataset.
                    batch_size  - Int.
                    shuffle     - Bool. Shuffle the data or not
                    num_workers - Int. How many thread you want to use to read the batch data
        """
        # Create loaders
        self.loaders = []
        for dataset in datasets:
            self.loaders.append(
                data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
            )
        
        # Check the sample method
        self.sample_method = None
        for dataset in datasets:
            if self.sample_method is None:
                self.sample_method = dataset.sample_method
            else:
                if self.sample_method != dataset.sample_method:
                    raise Exception("Sample methods are not consistant, {} <=> {}".format(
                        self.sample_method, dataset.sample_method
                    ))

        # Check the iteration number 
        self.iter_num = 0
        for i, dataset in enumerate(datasets):
            if i == 0:
                self.iter_num = len(dataset)
            else:
                if self.sample_method == UNDER_SAMPLING:
                    self.iter_num = min(self.iter_num, len(dataset))
                else:
                    self.iter_num = max(self.iter_num, len(dataset))
        self.iter_num = round(self.iter_num / batch_size)

    def __len__(self):
        return self.iter_num

    def __iter__(self):
        self.iter_loaders = []
        for loader in self.loaders:
            self.iter_loaders.append(iter(loader))
        return self

    def __next__(self):
        result = []
        for loader in self.iter_loaders:
            for _ in loader.__next__():
                result.append(_)
        return tuple(result)

class IterationLoader(Iterator):
    def __init__(self, loader, max_iter = 1):
        """
            Constructor of the loader with specific iteration (not epoch)
            The iteration object will create again while getting end
            
            Arg:    loader      - The torch.data.DataLoader object
                    max_iter    - The maximun iteration
        """
        super().__init__()
        self.loader = loader
        self.loader_iter = iter(self.loader)
        self.iter = 0
        self.max_iter = max_iter

    def __next__(self):
        try:
            result_tuple = next(self.loader_iter)
        except:
            self.loader_iter = iter(self.loader)
            result_tuple = next(self.loader_iter)
        self.iter += 1
        if self.iter <= self.max_iter:
            return result_tuple
        else:
            print("", end='')
            raise StopIteration()

    def __len__(self):
        return self.max_iter