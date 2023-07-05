""" Loss 损失函数 """

import torch.nn as nn

def loss(args):
    loss_lower = args.loss.lower()
    
    if loss_lower == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_lower == 'mse':
        loss = nn.MSELoss()
    elif loss_lower == 'l1':
        loss = nn.L1Loss()
    else:
        assert False and "Invalid optimizer"
    fp=open('output.log','a+')
    print(f'loss is {loss}', file=fp)
    fp.close()
    return loss
    
    
