import torch
import argparse

from network import MLP1, CNN1, CNN2, ResNet
from losses import loss
from optimizer import create_optimizer
from engine import train, test
from dataset import data_load
from plot import plot_loss_and_acc

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='CIFAR10', type=str,
                        help='choose dataset (default: CIFAR10)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=10)
    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)

    parser.add_argument('--loss', type=str, default='ce')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--exp', type=str, default='debug')
    
    parser.add_argument('--model', type=str, default='resnet')
    return parser


def main(args):
    # 设置训练设备
    device = torch.device(args.device)
    in_chans = (1 if args.dataset.lower() == 'mnist' else 3)
    img_size = (28 if args.dataset.lower() == 'mnist' else 32)
    print(img_size)
    # 定义模型
    if args.model == 'cnn1':
        model = CNN1(img_size=img_size ,in_chans=in_chans)
    elif args.model == 'cnn2':
        model = CNN2(img_size=img_size ,in_chans=in_chans)
    elif args.model == 'mlp1':
        model = MLP1(img_size=img_size ,in_chans=in_chans)
    elif args.model == 'resnet':
        model = ResNet()

    model = model.to(device)
    # 打印信息
    print(f"使用 {device} device")
    print(f"数据集:{args.dataset}")
    print(f"模型:{args.model}")    


    # 定义代价函数
    criterion = loss(args)
    # 定义优化器
    optimizer = create_optimizer(args, model)
    train_loader, test_loader = data_load(args)
    # 训练
    model_trained, los, acc = train(model=model, 
                                    criterion=criterion,
                                    train_loader=train_loader,
                                    optimizer=optimizer,
                                    device=device,
                                    max_epoch=args.epoch,
                                    disp_freq=100)
    # 测试
    test(model=model,
        criterion=criterion,
        test_loader=test_loader,
        device=device)
    
    # 模型保存
    # torch.save(model.state_dict(), 'model/my_model.pth')
    
    # 绘制损失和准确率图
    suffix = args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) +  '.png'
    path = ['los_acc/loss_' + suffix, 'los_acc/acc_' + suffix,]
    plot_loss_and_acc({'TRAIN': [los, acc]}, path)

if __name__ == "__main__":
    """
    Please design the initial and target state.
    """
    parser = argparse.ArgumentParser('Prombelm solver', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
