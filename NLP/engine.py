""" 训练和测试 """

import numpy as np
import torch


def train(model: torch.nn.Module, criterion,
        train_iter, val_iter, optimizer: torch.optim.Optimizer,  
        device:torch.device, max_epoch: int, disp_freq):  # para: val_set
    
	avg_train_loss, avg_train_acc = [], []
	avg_val_loss, avg_val_acc = [], []

	# Training process
	for epoch in range(max_epoch):
		batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, train_iter, optimizer, device,
													max_epoch, disp_freq, epoch)
		batch_val_loss, batch_val_acc = validate(model, criterion, val_iter)

		avg_train_acc.append(np.mean(batch_train_acc))
		avg_train_loss.append(np.mean(batch_train_loss))
		avg_val_acc.append(np.mean(batch_val_acc))
		avg_val_loss.append(np.mean(batch_val_loss))

		print('Epoch [{}]\t Average training loss {:.4f}\t Average training accuracy {:.4f}'.format(
			epoch, avg_train_loss[-1], avg_train_acc[-1]))

		print('Epoch [{}]\t Average validation loss {:.4f}\t Average validation accuracy {:.4f}'.format(
		 	epoch, avg_val_loss[-1], avg_val_acc[-1]))

	return model, avg_train_loss, avg_train_acc , avg_val_loss, avg_val_acc



def train_one_epoch(model: torch.nn.Module, criterion,
                    train_iter, optimizer: torch.optim.Optimizer,  
                    device:torch.device, max_epoch: int, disp_freq, epoch):
	model.train(True)
	batch_train_loss, batch_train_acc = [], []
	max_train_iteration = len(train_iter)
	counter = 0
	for batch in train_iter:
		# 获得数据和对应的标签
		counter += 1 # 训练次数+1
		text = batch.text
		label = batch.label - 1
		# 梯度清0
		optimizer.zero_grad()
		output = model(text)
		loss = criterion(output, label)
        # 交叉熵代价函数out(batch,C),labels(batch)
        # 计算梯度
		loss.backward()
        # 修改权值
		optimizer.step()
  		# 记录损失与精确度
		batch_train_loss.append(loss.item())
        # 获得最大值，以及最大值所在的位置
		_, predicted = torch.max(output, 1)
        # 预测正确的数量
		batch_train_acc.append(((predicted == label).sum() / len(predicted)).item())
		if counter % disp_freq == 0:
			print("Epoch [{}][{}]\t Batch [{}][{}]\t Training Loss {:.4f}\t Accuracy {:.4f}".format(
				epoch, max_epoch, counter, max_train_iteration,
				np.mean(batch_train_loss), np.mean(batch_train_acc)))
	return batch_train_loss, batch_train_acc


def validate(model, criterion, val_iter):
	batch_val_acc, batch_val_loss = [], []

	# 在验证集上测试模型
	val_loss = 0
	val_acc = 0
	model.eval()
	with torch.no_grad():
		for batch in val_iter:
			text = batch.text
			label = batch.label - 1
			output = model(text)
			loss = criterion(output, label)
			
			_, predicted = torch.max(output, 1)
			
			val_loss += loss.item()
			val_acc += (predicted == label).float().mean().item()
			batch_val_loss.append(loss.item())
			batch_val_acc.append((predicted == label).float().mean().item())
	val_loss /= len(val_iter)
	val_acc /= len(val_iter)
	print("Validate Loss {:.4f} Accuracy {:.4f}".format(np.mean(batch_val_loss), np.mean(batch_val_acc)))
	print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
	return batch_val_loss, batch_val_acc


def test(model, criterion, test_iter, device:torch.device):
	print('Testing...')
	model.train(False)
	batch_test_acc = 0
	test_loss = 0
	test_acc = 0

	for batch in test_iter:
		text = batch.text
		label = batch.label - 1
		# Only forward pass
		predictions = model(text)
		loss = criterion(predictions, label)
		_, predicted = torch.max(predictions, 1)

		test_loss += loss.item()
		test_acc += (predicted == label).float().mean().item()  
  
	test_loss /= len(test_iter)
	test_acc /= len(test_iter)

	print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
