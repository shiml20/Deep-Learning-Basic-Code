# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:55:31 2023

@author: Administrator
"""

import os
import math
import logging
import numpy as np
import argparse
import time
from functools import partial
from copy import Error, deepcopy
from collections import Counter,OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from sklearn.model_selection import train_test_split

from network import CNN, CNN2, CNN3, ResNet, VisionTransformer
from optimizer import create_optimizer
from losses import loss
from engine import test

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='Finance', type=str,
                        help='choose dataset (default: Finance)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size of the dataset default(128)')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epochs of training process default(10)')
    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--loss', type=str, default='ce',
                        help='define loss function (default: CrossEntropy)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='choose running device (default: Cuda)')
    parser.add_argument('--exp', type=str, default='debug',
                        help='choose using mode, (default: experiment mode)')
    parser.add_argument('--model', type=str, default='cnn',
                        help='choosing the model (default: cnn)')
    parser.add_argument('--infer', type=int, default=1,
                        help='if infer mode or not default(0)')
    parser.add_argument('--small_set', type=int, default=0,
                        help='using the small dataset or not default(0)')
    return parser

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

# 在测试集上的图像增强只做确定性的操作
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def main(args):
    # 加载训练集、验证集、测试集
    new_data_dir="dataset"
    # dataset = datasets.ImageFolder(root=os.path.join(new_data_dir, 'train'), transform=transform_train)
    test_ds = datasets.ImageFolder(root=os.path.join(new_data_dir, 'test'), transform=transform_train)
    # train_set, val_set = train_test_split(dataset, test_size=0.3, random_state=42)
    # Define the data loader for training data
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=True) 


    
    path = ["Finance_cnn3_lr1e-05_adam_wd0.0_epoch20_model.pth",
            # "pre-trained/Finance_cnn2_lr0.0005_adam_wd0.0001_epoch20_model.pth",
            # "pre-trained/Finance_cnn2_lr1e-05_adam_wd0.0_epoch20_model.pth",
            # "pre-trained/Finance_cnn2_lr1e-05_adam_wd0.0001_epoch20_model.pth",
            # "pre-trained/Finance_cnn2_lr5e-05_adam_wd0.0001_epoch20_model.pth",
            # "pre-trained/Finance_cnn3_lr1e-05_adam_wd0.0001_epoch1_model.pth",
            # "pre-trained/Finance_cnn3_lr1e-05_adam_wd0.0001_epoch5_model.pth",
            # "pre-trained/Finance_cnn2_lr0.0001_adam_wd0.0001_epoch20_model.pth"
            ]
    # 替换为你要获取名字的文件夹路径  
    folder_path = 'pre-trained'  
  
    # 获取文件夹中所有文件的名字  
    # path = os.listdir(folder_path)  
    print(path)
    for pth in path:
            # 设置训练设备
        device = torch.device(args.device)
        in_chans = 3
        # 定义模型
        if 'cnn1' in pth:
            model = CNN(in_chans=in_chans)
        elif 'cnn2' in pth:
            model = CNN2()
        elif 'cnn3' in pth:
            model = CNN3()
        elif 'resnet' in pth:
            model = ResNet(in_chans=in_chans, img_H=180, img_W=96)
        elif 'vit' in pth:
            model = VisionTransformer(img_size=32, patch_size=2, num_classes=10, num_heads=1, depth=1, embed_dim=32)        

        model = model.to(device)
        # 打印信息
        fp=open('inference.log','a+')
        print(f"using {device} device", file=fp)
        print(f"dataset:{args.dataset}", file=fp)
        print(f"model:{args.model}", file= fp)
        print(f"using {device} device")
        print(f"dataset:{args.dataset}")
        print(f"model:{args.model}")
        fp.close()
        if args.infer == 1:
            pre_trained_path = pth
            # prefix = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch)
            model.load_state_dict(torch.load(os.path.join(folder_path, pre_trained_path)))
            print("using pre-trained model" + pre_trained_path)
            fp=open('inference.log','a+')
            print("using pre-trained model" + pre_trained_path, file=fp)
            fp.close()
            
        
        # 定义代价函数
        criterion = loss(args)
        # 定义优化器
        optimizer = create_optimizer(args, model)
        
        # 测试
        test(model,
            criterion=criterion,
            test_loader=test_loader,
            device=device)

if __name__ == "__main__":
    """
    Main code
    """
    parser = argparse.ArgumentParser('Prombelm solver', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)