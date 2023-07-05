""" 训练和测试 """

import numpy as np
import torch
import sklearn

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
		if epoch % 100 == 0:
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
	(inputs, labels) = train_loader
	# for (inputs, labels) in enumerate(train_loader):
	# 获得数据和对应的标签
	inputs = inputs.to(device)
	labels = labels.to(device)
	# 获得模型预测结果，（64，10）
	
	output = model(inputs)
 
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


	(inputs, labels) = test_loader
	# for (inputs, labels) in enumerate(train_loader):
	# 获得数据和对应的标签
	inputs = inputs.to(device)
	labels = labels.to(device)
	# 获得模型预测结果，（64，10）
	
	logit = model(inputs)
 
	# 交叉熵代价函数out(batch,C),labels(batch)
	loss = criterion(logit, labels)

	# 获得最大值，以及最大值所在的位置
	_, predicted = torch.max(logit, 1)
	# 预测正确的数量
	batch_test_acc = (predicted == labels).sum() / len(predicted)
 
	predicted = predicted.cpu()
	labels = labels.cpu()
 
	micro_f1 = sklearn.metrics.f1_score(labels, predicted, average='micro')
	micro_precision = sklearn.metrics.precision_score(labels, predicted, average='macro')
	micro_recall_score = sklearn.metrics.recall_score(labels, predicted, average='micro')
 
	
	macro_f1 = sklearn.metrics.f1_score(labels, predicted, average='macro')
	macro_precision = sklearn.metrics.precision_score(labels, predicted, average='macro')
	macro_recall_score = sklearn.metrics.recall_score(labels, predicted, average='macro')
	acc = sklearn.metrics.accuracy_score(labels, predicted)
 
	print("The test accuracy is {}.\n".format(batch_test_acc))
	print("Micro_f1 {}, Micro_precision {}, Micro_recall_score {}, Acc {}".format(micro_f1, micro_precision, micro_recall_score, acc))
	print("Macro_f1 {}, Macro_precision {}, Macro_recall_score {}, Acc {}".format(macro_f1, macro_precision, macro_recall_score, acc))
