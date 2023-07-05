from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context

def data_set(args):
    # 数据集的预处理
    data_preprocess = transforms.Compose([
        transforms.Resize(112, interpolation=3),
        transforms.CenterCrop(112),
        transforms.ToTensor(), # 对原有数据转成Tensor类型
        # transforms.Normalize([0.5],[0.5])
        ]) # 用平均值和标准偏差归一化张量图像
    
    data_set_lower = args.dataset.lower()
    root = '100-bird-species/'
    train_dataset =  datasets.ImageFolder(root + 'train', transform=data_preprocess)
    val_dataset =  datasets.ImageFolder(root + 'valid', transform=data_preprocess)
    test_dataset = datasets.ImageFolder(root + 'test', transform=data_preprocess)

    return train_dataset, val_dataset, test_dataset


def data_load(args):
    # 装载训练集
    train_dataset, val_dataset, test_dataset = data_set(args)
    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
    # 装载验证集
    val_loader = DataLoader(dataset=val_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
    # 装载测试集
    test_loader = DataLoader(dataset=test_dataset,
                        batch_size=args.batch_size,
                        shuffle=True)
    return train_loader, val_loader, test_loader