""" Net 网络结构 """

import torch.nn as nn


# 定义网络结构
class MLP1(nn.Module):
    def __init__(self, img_size=32, in_chans=3):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(img_size*img_size*in_chans, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        # ([64, 1, 28, 28])->(64,784)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class CNN1(nn.Module):
    def __init__(self, img_size=32, in_chans=3):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        # 由于MaxPooling降采样两次，特征图缩减为原图的 1/4 * 1/4
        self.fc1 = nn.Sequential(nn.Linear(in_features=64*int(img_size/4)*int(img_size/4), out_features=1000), nn.Dropout(p=0.4), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=10), nn.Softmax(dim=1))
        
    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        x = self.conv1(x)
        x = self.conv2(x)
        # 合并后两维通道, 保留第一维(batch_size)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class CNN2(nn.Module):
    def __init__(self, img_size=32, in_chans=3):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        # 由于MaxPooling降采样两次，特征图缩减为原图的 1/4 * 1/4
        self.fc1 = nn.Sequential(nn.Linear(in_features=128*int(img_size/4)*int(img_size/4), out_features=1000), nn.Dropout(p=0.4), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=1000, out_features=10), nn.Softmax(dim=1))
        
    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        x = self.conv1(x)
        x = self.conv2(x)
        # 合并后两维通道, 保留第一维(batch_size)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, img_H=32, img_W=32, in_chans=3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2), nn.ReLU())
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # 由于MaxPooling降采样两次，特征图缩减为原图的 1/4 * 1/4
        self.fc1 = nn.Sequential(nn.Linear(in_features=64*int(img_H/4)*int(img_W/4), out_features=100), nn.Dropout(p=0.4), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=100, out_features=10), nn.Softmax(dim=1))



    def forward(self, x):
        # mnist: ([batch_size, 1, 28, 28]) / cifar10: ([batch_size, 3, 32, 32])
        # print(x.shape)
        # print(self.conv1(x).shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        # print(self.conv2(x).shape)
        shortcut = x        
        x = self.conv2(x)
        x += shortcut
        x = self.maxpool2(x)
        # print(x.shape)
        # 合并后两维通道, 保留第一维(batch_size)
        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        return x