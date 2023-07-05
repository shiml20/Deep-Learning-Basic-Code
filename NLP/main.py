import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.datasets import SST
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
import time
import torch.nn.functional as F
from network import LSTM
import argparse
import numpy as np
from losses import loss
from optimizer import create_optimizer
from engine import train, test, validate
from dataset import data_load
from plot import plot_loss_and_acc

import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('CNN Training', add_help=False)
    parser.add_argument('--dataset', default='SST', type=str,
                        help='choose dataset (default: SST)')
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
    
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--tag', type=str, default='debug')
    
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrained', type=int, default=1)
    
    return parser

def main(args):
    # set up fields
    TEXT = Field(fix_length=60)
    LABEL = Field(sequential=False,dtype=torch.long)

    # make splits for data
    # DO NOT MODIFY: fine_grained=True, train_subtrees=False
    train_data, val_data, test_data = SST.splits(
    TEXT, LABEL, fine_grained=True, train_subtrees=False)

    # print information about the data
    print('train.fields', train_data.fields)
    print('len(train)', len(train_data))
    print('vars(train[0])', vars(train_data[0]))

    # build the vocabulary
    # you can use other pretrained vectors, refer to https://github.com/pytorch/text/blob/master/torchtext/vocab.py
    TEXT.build_vocab(train_data, vectors=Vectors(name='vector.txt', cache='./data'))
    LABEL.build_vocab(train_data)
    # We can also see the vocabulary directly using either the stoi (string to int) or itos (int to string) method.
    print("itos", TEXT.vocab.itos[:10])
    print("stoi", LABEL.vocab.stoi)
    print("most_common", TEXT.vocab.freqs.most_common(20))

    # print vocab information
    print('len(TEXT.vocab)', len(TEXT.vocab))
    print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

    batch_size = 64
    # make iterator for splits
    train_it, val_it, test_it = BucketIterator.splits(
    (train_data, val_data, test_data), batch_size=batch_size,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 初始化模型
    vocab_size = len(TEXT.vocab)
    # print(vocab_size)
    # time.sleep(100)
    embedding_dim = 300
    hidden_dim = args.hidden_dim
    output_dim = 5
    num_layers = args.num_layers
    dropout = args.dropout
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

    # Copy the pre-trained word embeddings we loaded earlier into the embedding layer of our model.
    pretrained_embeddings = TEXT.vocab.vectors

    print(pretrained_embeddings.shape)

    # you should maintain a nn.embedding layer in your network
    if args.pretrained:
        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.embedding.requires_grad = False
    # 设置训练设备
    device = torch.device(args.device)
    # 定义模型
    model = model.to(device)
    # 打印信息
    print(f"使用 {device} device")
    print(f"数据集:{args.dataset}")
    print(f"模型:{args.model}")    

    # 定义代价函数
    criterion = loss(args)
    # 定义优化器
    optimizer = create_optimizer(args, model)
    # 训练
    model_trained, train_los, train_acc, val_los, val_acc = train(model=model, 
                                    criterion=criterion,
                                    train_iter=train_it,
                                    val_iter=val_it,
                                    optimizer=optimizer,
                                    device=device,
                                    max_epoch=args.epoch,
                                    disp_freq=100)
    # 测试
    test(model=model,
        criterion=criterion,
        test_iter=test_it,
        device=device)
    
    # 模型保存
    # torch.save(model.state_dict(), 'model/my_model.pth')
    
    # 绘制损失和准确率图
    suffix1 = args.tag + args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) +  '.png'
    path1 = ['los_acc/train_loss_' + suffix1, 'los_acc/train_acc_' + suffix1]
    suffix2 = args.tag + args.dataset + '_' + args.model + '_lr' + str(args.lr) + '_' + str(args.opt) + '_wd' + str(args.weight_decay) + '_epoch' + str(args.epoch) +  '.png'
    path2 = ['los_acc/val_loss_' + suffix2, 'los_acc/val_acc_' + suffix2]   
    plot_loss_and_acc({'TRAIN': [train_los, train_acc]}, path1)
    plot_loss_and_acc({'VAL': [val_los, val_acc]}, path2)

if __name__ == "__main__":
    """
    Please design the initial and target state.
    """
    parser = argparse.ArgumentParser('Prombelm solver', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
