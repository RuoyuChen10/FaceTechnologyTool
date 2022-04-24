# -*- coding: utf-8 -*-  

"""
Created on 2022/1/11

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np
import copy
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict

from dataset import VGGFace2Dataset
from models.iresnet_edl import iresnet50, iresnet100
from models.metrics import ArcFace, CosFace, FocalLoss
from models.evidential import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence, get_device, one_hot_embedding

from tqdm import tqdm

class INFO():
    def __init__(self, save_log):
        self.save_log=save_log
        if os.path.exists(save_log):
            os.remove(save_log)

    def __call__(self, string):
        print(string)
        if self.save_log != None:
            with open(self.save_log,"a") as file:
                file.write(string)
                file.write("\n")


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

def train_model(
    model,
    metric,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    info = None,
    print_freq = 10
):

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "epoch": []}
    accuracy = {"accuracy": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}

    for epoch in range(num_epochs):
        info("Epoch {}/{}".format(epoch, num_epochs - 1))
        info("-" * 10)

        # Each epoch has a training and validation phase
        info("Training...")
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0.0
        correct = 0

        # Iterate over data.
        for i, (inputs, labels) in enumerate(dataloaders):

            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device).long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # outputs = metric(outputs, labels)

                _, preds = torch.max(outputs, 1)
                loss = criterion(
                    outputs, y.float(), epoch, num_classes, 10, device
                )
                print(loss)
                match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            iters = epoch * len(dataloaders) + i

            if acc>0.9:
                torch.save(model, "./checkpoint/evidential/model-high-acc-"+str(acc)+'.pth')

            # View the train process
            if iters % print_freq == 0:
                # outputs = outputs.data.cpu().numpy()
                # output = np.argmax(outputs, axis=1)
                # labels = labels.data.cpu().numpy()

                # acc = np.mean((output==labels).astype(int))
                speed = print_freq / (time.time() - since)
                time_str = time.asctime(time.localtime(time.time()))

                info("{} train epoch {} iter {} {:.4f} iters/s loss {:.4f} acc {}".format(time_str, epoch, i, speed, loss.item(), acc))

        if scheduler is not None:
            scheduler.step()

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        losses["loss"].append(epoch_loss)
        losses["epoch"].append(epoch)
        accuracy["accuracy"].append(epoch_acc.item())
        accuracy["epoch"].append(epoch)

        info(
            "training epoch {} loss: {:.4f} acc: {:.4f}".format(
                epoch, epoch_loss, epoch_acc
            )
        )

        # Save model
        if epoch % 1 == 0:
            torch.save(model, "./checkpoint/evidential/model-item-epoch-"+str(epoch)+'.pth')

    time_elapsed = time.time() - since
    info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics

def main(args):
    """
    Train the network
    """
    device = torch.device(args.device)

    info = INFO(args.save_log)

    # Dataloader
    train_dataset = VGGFace2Dataset(dataset_root=args.dataset_root,dataset_list=args.train_list)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)

    model = define_backbone(args.backbone, args.num_classes)
    metric = define_metric(args.metric)      # ArcFace etc.

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
    metric.to(device)

    # Loss
    if args.uncertain == "log":
        criterion = edl_log_loss
    elif args.uncertain == "digamma":
        criterion = edl_digamma_loss
    elif args.uncertain == "mse":
        criterion = edl_mse_loss

    # optimizer
    if args.opt == "sgd":
        optimizer = torch.optim.SGD([{'params':model.parameters()}],
                lr = args.lr, weight_decay = 0.01)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam([{'params':model.parameters()}],
                lr = args.lr, weight_decay = 0.01)

    exp_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    model, metrics = train_model(
            model,
            metric,
            dataloaders=train_loader,
            num_classes=args.num_classes,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=args.epoch,
            device=device,
            info = info,
            print_freq = args.print_freq
        )
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train paramters.')
    # general
    parser.add_argument('--dataset-root', type=str,
                        default='/home/cry/data2/VGGFace2/train_align_arcface',
                        help='')
    parser.add_argument('--train-list', type=str,
                        default='./data/seed1/train.txt',
                        help='')
    parser.add_argument('--val-list', type=str,
                        default='./data/val.txt',
                        help='')
    parser.add_argument('--num-classes', type=int,
                        # default=8615,
                        default=8631,
                        help='')
    parser.add_argument('--backbone', type=str,
                        default='resnet50',
                        choices=['resnet50','resnet100'],
                        help='')
    parser.add_argument('--pre-trained', type=str,
                        default='./pretrained/ms1mv3_arcface_r50_fp16/backbone.pth',
                        help='')
    parser.add_argument('--metric', type=str,
                        default='arcface',
                        choices=['arcface','cosface'],
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=128,
                        help='')
    parser.add_argument('--epoch', type=int,
                        default=50,
                        help='')
    parser.add_argument('--lr', type=float,
                        default=1e-5,
                        help='')
    parser.add_argument('--opt', type=str,
                        default='adam',
                        choices=['sgd','adam'],
                        help='')
    parser.add_argument('--uncertain', type=str,
                        default='log',
                        choices=['log','mse','digamma'],
                        help='')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda','cpu'],
                        help='')
    parser.add_argument('--print-freq', type=int,
                        default=50,
                        help='')
    parser.add_argument('--save-log', type=str,
                        default='./train-evidential.log',
                        help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)