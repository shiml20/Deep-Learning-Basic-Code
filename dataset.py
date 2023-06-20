from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context

def data_set(args):
    # 数据集的预处理
    data_preprocess = transforms.Compose([
        transforms.ToTensor(), # 对原有数据转成Tensor类型
        transforms.Normalize([0.5],[0.5])]) # 用平均值和标准偏差归一化张量图像
    
    data_set_lower = args.dataset.lower()
    
    if (data_set_lower == 'cifar10'):
        train_dataset = datasets.CIFAR10(root='./',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
        # 下载测试集
        test_dataset = datasets.CIFAR10(root='./',
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)
        # 下载训练集
    elif (data_set_lower == 'mnist'):
        train_dataset = datasets.MNIST(root='./',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)
        # 下载测试集
        test_dataset = datasets.MNIST(root='./',
                       train=False,
                       transform=transforms.ToTensor(),
                       download=True)
    return train_dataset, test_dataset


def data_load(args):
    # 装载训练集
    train_dataset, test_dataset = data_set(args)
    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
    # 装载测试集
    test_loader = DataLoader(dataset=test_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
    return train_loader, test_loader