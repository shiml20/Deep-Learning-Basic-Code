import os
from collections import Counter 
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import argparse
from network import ResNet, VisionTransformer
from losses import loss
from optimizer import create_optimizer
from engine import train, test
from plot import plot_loss_and_acc
from dataset import data_load
import time
from sklearn.model_selection import train_test_split
from vit import VisionTransformer
# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='choose dataset (default: Finance)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size of the dataset default(128)')
    parser.add_argument('--epoch', type=int, default=2,
                        help='epochs of training process default(10)')
    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--loss', type=str, default='ce',
                        help='define loss function (default: CrossEntropy)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='choose running device (default: Cuda)')
    parser.add_argument('--exp', type=str, default='debug',
                        help='choose using mode, (default: experiment mode)')
    parser.add_argument('--model', type=str, default='resnet',
                        help='choosing the model (default: cnn)')
    parser.add_argument('--infer', type=int, default=0,
                        help='if infer mode or not default(0)')
    parser.add_argument('--small_set', type=int, default=0,
                        help='using the small dataset or not default(0)')
    return parser



transform_train = transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和宽均为224像素的新图像
    # transforms.RandomResizedCrop(224, scale=(0.08, 1.0),  
    #                              ratio=(3.0/4.0, 4.0/3.0)),
    # # 以0.5的概率随机水平翻转
    # transforms.RandomHorizontalFlip(),
    # # 随机更改亮度、对比度和饱和度
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    # transforms.Resize(img_resize),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    # transforms.CenterCrop(img_resize),
    transforms.ToTensor(),
    # 对各个通道做标准化，(0.485, 0.456, 0.406)和(0.229, 0.224, 0.225)是在ImageNet上计算得的各通道均值与方差
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet上的均值和方差
])

# 在测试集上的图像增强只做确定性的操作
transform_test = transforms.Compose([
    # transforms.Resize(img_resize),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    # transforms.CenterCrop(img_resize),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



def main(args):
    #然后开始加载数据集
    new_data_dir="dataset"
    # 设置训练设备
    device = torch.device(args.device)
    in_chans = 3
    # 定义模型
    if args.model == 'resnet':
        model = ResNet(in_chans=in_chans, img_H=224, img_W=224)
    elif args.vit == 'vit':
        model = VisionTransformer(img_size=224, patch_size=16, num_classes=525, num_heads=2, depth=12, embed_dim=384)        

    model = model.to(device)
    # 打印信息
    fp=open('output.log','a+')
    print(f"using {device} device", file=fp)
    print(f"dataset:{args.dataset}", file=fp)
    print(f"model:{args.model}", file= fp)
    print(f"using {device} device")
    print(f"dataset:{args.dataset}")
    print(f"model:{args.model}")
    fp.close()
    

    train_loader, test_loader = data_load(args)
    val_loader = test_loader
    if args.infer == 1:
        model.load_state_dict(torch.load('my_model.pth'))
    
    # 定义代价函数
    criterion = loss(args)
    # 定义优化器
    optimizer = create_optimizer(args, model)
    
    # 训练
    if not args.infer:
        model_trained, best_model, train_los, train_acc, val_los, val_acc = train(model=model, 
                                        criterion=criterion,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        optimizer=optimizer,
                                        device=device,
                                        max_epoch=args.epoch,
                                        disp_freq=100)
    # 测试
    test(model=best_model,
        criterion=criterion,
        test_loader=test_loader,
        device=device)
    
    # 绘制损失和准确率图
    fp=open('output.log','a+')
    print(f'Drawing...', file=fp)
    print(f'Drawing...')
    
    # 绘制损失和准确率图
    suffix1 = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) +  '.png'
    path1 = ['train_loss_' + suffix1, 'train_acc_' + suffix1]
    suffix2 = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) +  '.png'
    path2 = ['val_loss_' + suffix2, 'val_acc_' + suffix2]   
    prefix = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch)
    if not args.infer:
        # 模型保存
        torch.save(best_model.state_dict(), prefix + '_model.pth')
        plot_loss_and_acc({'TRAIN': [train_los, train_acc]}, path1)
        plot_loss_and_acc({'VAL': [val_los, val_acc]}, path2)
        print("Draw Done", file=fp)
        print("Draw Done")
    fp.close()

if __name__ == "__main__":
    """
    Main code
    """
    parser = argparse.ArgumentParser('Prombelm solver', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
