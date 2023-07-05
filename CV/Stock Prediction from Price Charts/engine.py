""" 训练和测试 """

import numpy as np
import torch
import time
import copy

def train(model: torch.nn.Module, criterion,
        train_loader, val_loader, optimizer: torch.optim.Optimizer,  
        device:torch.device, max_epoch: int, disp_freq):  # para: val_set
    
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []

	min_val_loss = 1e9  #保证第一次一定会更新
	last_min_ind = -1
	early_stopping_epoch = 5    #early stop
	best_val_acc = 0
	# Training process
	for epoch in range(max_epoch):
		batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, train_loader, optimizer, device,
													max_epoch, disp_freq, epoch)
		batch_val_loss, batch_val_acc = validate(model, criterion, val_loader, device)

		avg_train_acc.append(np.mean(batch_train_acc))
		avg_train_loss.append(np.mean(batch_train_loss))
		avg_val_acc.append(np.mean(batch_val_acc))
		avg_val_loss.append(np.mean(batch_val_loss))
		fp=open('output.log','a+')
		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]), file=fp)
		print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
			epoch, avg_val_loss[-1], avg_val_acc[-1]), file=fp)
		fp.close()

		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]))
		print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
			epoch, avg_val_loss[-1], avg_val_acc[-1]))

		final_epoch = 0
		running_loss = avg_val_loss[-1]

		if avg_val_acc[-1] > best_val_acc:
			best_val_acc = avg_val_acc[-1]
			best_model = copy.deepcopy(model)
			print("Best ckpt", best_val_acc)

		#early stop
		if running_loss < min_val_loss:
			last_min_ind = epoch
			min_val_loss = running_loss    #检测每一次epoch之后val loss有没有变得更小
		elif epoch - last_min_ind >= early_stopping_epoch:
			final_epoch = epoch
			break

	return model, best_model, avg_train_loss, avg_train_acc ,avg_val_loss, avg_val_acc



def train_one_epoch(model: torch.nn.Module, criterion,
                    train_loader, optimizer: torch.optim.Optimizer,  
                    device:torch.device, max_epoch: int, disp_freq, epoch):
	model.train(True)
	batch_train_loss, batch_train_acc = [], []
	max_train_iteration = len(train_loader)
 

	for iteration, (inputs, labels) in enumerate(train_loader):
		start_time = time.time()
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
		end_time = time.time()
		t = -start_time + end_time
		if iteration % disp_freq == 0:
			fp=open('output.log','a+')
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}\t Time(Iter) {:.4f}".format(
				epoch, max_epoch, iteration, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc), t),
         		file=fp)
			fp.close()
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}\t Time(Iter) {:.4f}".format(
				epoch, max_epoch, iteration, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc), t))
	return batch_train_loss, batch_train_acc


def validate(model, criterion, val_loader, device:torch.device):
	batch_val_acc, batch_val_loss = [], []
	model.train(False)

	for iteration, (inputs, labels) in enumerate(val_loader):
		# Get validating data and label
		inputs = inputs.to(device)
		labels = labels.to(device)

		# Only forward pass
		logit = model(inputs)
		loss = criterion(logit, labels)
		_, predicted = torch.max(logit, 1)
		# Record loss and accuracy
		batch_val_loss.append(loss.item())
		batch_val_acc.append(((predicted == labels).sum() / len(predicted)).item())

	return batch_val_loss, batch_val_acc


def test(model, criterion, test_loader, device:torch.device):
	fp=open('output.log','a+')
	print('Testing...', file=fp)
	fp.close()
	model.train(False)
	batch_test_acc = 0
	TP,TN,FP,FN = 0,0,0,0
	P,N = 0,0
	sum = 0
	for iteration, (inputs, labels) in enumerate(test_loader):
		inputs = inputs.to(device)
		labels = labels.to(device)
		# Only forward pass
		logit = model.forward(inputs)
		loss = criterion.forward(logit, labels)

		_, predicted = torch.max(logit, 1)
        # 预测正确的数量
		# batch_test_acc += (predicted == labels).sum()
		
		batch_test_acc += (predicted == labels[:,-1,:].squeeze(1)).sum()
		TP += ((predicted == labels)*(labels == 1)).sum()
		TN += ((predicted == labels)*(labels == 0)).sum()
		FP += ((predicted != labels)*(labels == 0)).sum()
		FN += ((predicted != labels)*(labels == 1)).sum()
		sum += len(predicted)
		P += TP+FP
		N += TN+FN
		if iteration % 100 == 0:
			fp=open('output.log','a+')
			print("{}: The batch accuracy is {}.".format(iteration, batch_test_acc/sum), file=fp)
			print("{}: The batch accuracy is {}.".format(iteration, batch_test_acc/sum))
	fp=open('inference.log','a+')
	print("The test accuracy is {}.\n".format(batch_test_acc/sum), file=fp)
	print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)), file=fp)
	fp.close()
	print("The test accuracy is {}.\n".format(batch_test_acc/sum))
	print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)))

            
    