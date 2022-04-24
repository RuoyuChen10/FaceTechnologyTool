# -*- coding: utf-8 -*-  

"""
Created on 2021/12/21

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

from dataset import VGGFace2Dataset
from models.iresnet import iresnet50, iresnet100
from models.metrics import ArcFace, CosFace, FocalLoss
from utils import INFO

from tqdm import tqdm

def define_backbone(net_type, nun_classes):
    """
    define the model
    """
    if net_type == "resnet50":
        backbone = iresnet50(nun_classes)
    elif net_type == "resnet100":
        backbone = iresnet100(nun_classes)
    return backbone

def define_metric(metric_type):
    if metric_type == "arcface":
        metric = ArcFace()
    elif metric_type == "cosface":
        metric = CosFace()
    return metric

def eval_model(backbone, validation_loader, device):
    backbone.eval()

    acc = 0

    with torch.no_grad():
        for ii, (data,label) in enumerate(validation_loader):
            print(ii/len(validation_loader))
            data = data.to(device)
            label = label.to(device).long()
            
            output = backbone(data)

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()

            acc += np.sum((output==label).astype(int))
            print(acc)

    acc = acc / len(validation_loader.dataset)
    print(acc)
    return acc

def main(args):
    """
    Train the network
    """
    device = torch.device(args.device)

    info = INFO(args.save_log)

    # Dataloader
    validation_dataset = VGGFace2Dataset(dataset_root=args.dataset_root,dataset_list=args.train_list)
    # validation_dataset = CelebADataset(dataset_root=args.dataset_root,dataset_list=args.val_list)

    validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=False)
    # validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True)

    model = define_backbone(args.backbone, args.num_classes)
    # model = torch.load(args.pre_trained)
    # metric = define_metric(args.metric)      # ArcFace etc.

    if args.pre_trained is not None and os.path.exists(args.pre_trained):
        # No related
        model_dict = model.state_dict()
        pretrained_param = torch.load(args.pre_trained)
        try:
            pretrained_param = pretrained_param.state_dict()
        except:
            pass

        new_state_dict = OrderedDict()
        for k, v in pretrained_param.items():
            if k in model_dict:
                new_state_dict[k] = v
                info("Load parameter {}".format(k))
            elif k[7:] in model_dict:
                new_state_dict[k[7:]] = v
                info("Load parameter {}".format(k[7:]))

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)
        info("Success load pre-trained model {}".format(args.pre_trained))

    # Set to device
    model.to(device)
    model.eval()
    # metric.to(device)

    # Multi GPU
    if args.device == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    acc = eval_model(model, validation_loader, device)
    info(str(acc))


def parse_args():
    parser = argparse.ArgumentParser(description='Train paramters.')
    # general
    parser.add_argument('--dataset-root', type=str,
                        default='/exdata2/RuoyuChen/Datasets/VGGFace2/train_align_arcface',
                        help='')
    parser.add_argument('--train-list', type=str,
                        default='./data/seed1-100/test.txt',
                        help='')
    parser.add_argument('--num-classes', type=int,
                        # default=8631,
                        default=100,
                        help='')
    parser.add_argument('--backbone', type=str,
                        default='resnet50',
                        choices=['resnet50','resnet100'],
                        help='')
    parser.add_argument('--pre-trained', type=str,
                        # default='./pretrained/ms1mv3_arcface_r50_fp16/backbone.pth',
                        default = "./checkpoint/vggface2_100_softmax/backbone-item-epoch-1.pth",
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=800,
                        help='')
    parser.add_argument('--print-freq', type=int,
                        default=50,
                        help='')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda','cpu'],
                        help='')
    parser.add_argument('--save-log', type=str,
                        default='./test-vggface2-100-process.log',
                        help='')
    parser.add_argument('--gpu-device', type=str, default="0,1",
                        help='GPU device')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    main(args)