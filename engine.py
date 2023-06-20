""" 训练和测试 """

import numpy as np
import torch


def train(model: torch.nn.Module, criterion,
        train_loader, optimizer: torch.optim.Optimizer,  
        device:torch.device, max_epoch: int, disp_freq):  # para: val_set
    
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []

	# Training process
	for epoch in range(max_epoch):
		batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, train_loader, optimizer, device,
													max_epoch, disp_freq, epoch)
		# batch_val_loss, batch_val_acc = validate(model, criterion, val_set, batch_size)

		avg_train_acc.append(np.mean(batch_train_acc))
		avg_train_loss.append(np.mean(batch_train_loss))
		# avg_val_acc.append(np.mean(batch_val_acc))
		# avg_val_loss.append(np.mean(batch_val_loss))

		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]))

		# print('Epoch [{}]\t Average validat,ion loss {:.4f}\t Average validation accuracy {:.4f}'.format(
		# 	epoch, avg_val_loss[-1], avg_val_acc[-1]))

	return model, avg_train_loss, avg_train_acc #, avg_val_loss, avg_val_acc



def train_one_epoch(model: torch.nn.Module, criterion,
                    train_loader, optimizer: torch.optim.Optimizer,  
                    device:torch.device, max_epoch: int, disp_freq, epoch):
	model.train(True)
	batch_train_loss, batch_train_acc = [], []
	max_train_iteration = len(train_loader)
 
	for iteration, (inputs, labels) in enumerate(train_loader):
        # 获得数据和对应的标签
		inputs = inputs.to(device)
		labels = labels.to(device)
		# 获得模型预测结果，（64，10）
		print(labels.shape)

  
		output = model(inputs)
		print(output.shape)
		print(labels.shape)
        # 交叉熵代价函数out(batch,C),labels(batch)
		loss = criterion(output, labels)
        # 梯度清0
		optimizer.zero_grad()
        # 计算梯度
		loss.backward()
        # 修改权值
		optimizer.step()
  		# 记录损失与精确度
		batch_train_loss.append(loss.item())
        # 获得最大值，以及最大值所在的位置
		_, predicted = torch.max(output, 1)
        # 预测正确的数量
		batch_train_acc.append(((predicted == labels).sum() / len(predicted)).item())
		if iteration % disp_freq == 0:
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
				epoch, max_epoch, iteration, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc)))
	return batch_train_loss, batch_train_acc


def validate(model, criterion, val_set, batch_size):
	batch_val_acc, batch_val_loss = [], []
	max_val_iteration = val_set.num_examples // batch_size

	for iteration in range(max_val_iteration):
		# Get validating data and label
		val_x, val_y = val_set.next_batch(batch_size)

		# Only forward pass
		logit = model.forward(val_x)
		loss = criterion.forward(logit, val_y)

		# Record loss and accuracy
		batch_val_loss.append(criterion.loss)
		batch_val_acc.append(criterion.acc)

	return batch_val_loss, batch_val_acc


def test(model, criterion, test_loader, device:torch.device):
	print('Testing...')
	model.train(False)
	batch_test_acc = 0

	for iteration, (inputs, labels) in enumerate(test_loader):
		inputs = inputs.to(device)
		labels = labels.to(device)
		# Only forward pass
		logit = model.forward(inputs)
		loss = criterion.forward(logit, labels)

		_, predicted = torch.max(logit, 1)
        # 预测正确的数量
		batch_test_acc = (predicted == labels).sum() / len(predicted)

	print("The test accuracy is {}.\n".format(batch_test_acc))
