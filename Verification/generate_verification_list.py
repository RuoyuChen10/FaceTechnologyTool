# -*- coding: utf-8 -*-  

"""
Created on 2021/4/5

@author: Ruoyu Chen
"""

import numpy as np
import argparse
import os
import random

from utils import *

def main(args):
    if args.Datasets_type == "VGGFace2-train":
        if os.path.isfile(VGGFace2_train):
            with open(VGGFace2_train, "r") as f:     # open the file
                data = f.read().split("\n")         # read file
        else:
            raise ValueError("File {} not in path".format(VGGFace2_train))
        datasets = []
        # choose pairs different
        i = 0
        while(True):
            data1 = random.choice(data)
            data2 = random.choice(data)
            if data1.split('/')[0] != data2.split('/')[0]:
                datasets.append([data1,data2,0])
                i+=1
            if i == 2*args.Pairs:
                break
        # choose pairs same identity
        identity = os.listdir(VGGFace2_train_image_path)
        i = 0
        while(True):
            id_ = random.choice(identity)
            choice_images = os.listdir(os.path.join(VGGFace2_train_image_path,id_))
            data1 = random.choice(choice_images)
            data2 = random.choice(choice_images)
            if data1 != data2:
                datasets.append([os.path.join(id_,data1),os.path.join(id_,data2),1])
                i+=1
            if i == args.Pairs:
                break
        
        np.savetxt('./text/VGGFace2-train.txt',
                   np.array(datasets),
                   fmt='%s %s %s',
                   delimiter='\t')
    elif args.Datasets_type == "VGGFace2-test":
        if os.path.isfile(VGGFace2_test):
            with open(VGGFace2_test, "r") as f:     # open the file
                data = f.read().split("\n")         # read file
        else:
            raise ValueError("File {} not in path".format(VGGFace2_test))
        datasets = []
        # choose pairs different
        i = 0
        while(True):
            data1 = random.choice(data)
            data2 = random.choice(data)
            if data1.split('/')[0] != data2.split('/')[0]:
                datasets.append([data1,data2,0])
                i+=1
            if i == 2*args.Pairs:
                break
        # choose pairs same identity
        identity = os.listdir(VGGFace2_test_image_path)
        i = 0
        while(True):
            id_ = random.choice(identity)
            choice_images = os.listdir(os.path.join(VGGFace2_test_image_path,id_))
            data1 = random.choice(choice_images)
            data2 = random.choice(choice_images)
            if data1 != data2:
                datasets.append([os.path.join(id_,data1),os.path.join(id_,data2),1])
                i+=1
            if i == args.Pairs:
                break
        
        np.savetxt('./text/VGGFace2-test.txt',
                   np.array(datasets),
                   fmt='%s %s %s',
                   delimiter='\t')
    elif args.Datasets_type == "Celeb-A":
        if os.path.isfile(CelebA):
            with open(CelebA, "r") as f:     # open the file
                data = f.read().split("\n")         # read file
        else:
            raise ValueError("File {} not in path".format(CelebA))
        datasets = []
        # choose pairs different
        for i in range(2*args.Pairs):
            a = random.choice(data)
            b = random.choice(data)
            if a.split(' ')[1] == b.split(' ')[1]:
                ver = 1
            else:
                ver = 0
            datasets.append([a.split(' ')[0],b.split(' ')[0],ver])
        # choose pairs same
        datas = []
        for i in data:
            datas.append([i.split(' ')[0],i.split(' ')[1]])
        datas = np.array(datas)

        i=0
        while(True):
            id_ = random.choice(data).split(' ')[1]
            indexs = np.argwhere(datas==id_)
            if len(indexs)>=2:
                index = random.sample(list(indexs),2)
            else:
                continue
            a = datas[index[0][0]][0]
            b = datas[index[1][0]][0]
            datasets.append([a,b,1])
            i = i+1
            if i == args.Pairs:
                break

        np.savetxt('./text/CelebA.txt',
                   np.array(datasets),
                   fmt='%s %s %s',
                   delimiter='\t')
    elif args.Datasets_type == "LFW":
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Verification List')
    # general
    parser.add_argument('--Datasets-type',
                        type=str,
                        default='VGGFace2-train',
                        choices=['VGGFace2-train','VGGFace2-test','Celeb-A','LFW'],
                        help='Which datasets using for face verification.')
    parser.add_argument('--Pairs',
                        type=int,
                        default=1000,
                        help='Face identity recognition network.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)