# -*- coding: utf-8 -*-  

"""
Created on 2021/05/11

@author: Ruoyu Chen
"""
import os
import random
import numpy as np

import torchvision.transforms as transforms

from PIL import Image
from torch.utils import data

class VGGFace2Dataset(data.Dataset):
    """
    Read datasets
    """
    def __init__(self, dataset_root, dataset_list):
        self.dataset_root = dataset_root
        with open(dataset_list,"r") as file:
            datas = file.readlines()

        data = [os.path.join(self.dataset_root, data_) for data_ in datas]

        self.data = np.random.permutation(data)
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        # Sample
        sample = self.data[index]
        
        # data and label information
        splits = sample.split(' ')
        image_path = splits[0]

        data = Image.open(image_path)
        data = self.transforms(data)
        label = np.int32(splits[1])

        return data.float(), label

class CelebADataset(data.Dataset):
    """
    Read datasets
    """
    def __init__(self, dataset_root, dataset_list):
        self.dataset_root = dataset_root
        with open(dataset_list,"r") as file:
            datas = file.readlines()

        data = [os.path.join(self.dataset_root, data_) for data_ in datas]

        self.data = np.random.permutation(data)
        self.transforms = transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        # Sample
        sample = self.data[index]
        
        # data and label information
        splits = sample.split(' ')
        image_path = splits[0]

        data = Image.open(image_path)
        data = self.transforms(data)
        label = np.int32(splits[1])

        return data.float(), label