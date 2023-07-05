""" Optimizer 优化器 """

import torch.optim as optim

def create_optimizer(args, model):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    parameters = model.parameters()

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = optim.Adadelta(parameters, **opt_args)
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    else:
        assert False and "Invalid optimizer"

    print("optimizer is", optimizer)

    return optimizer