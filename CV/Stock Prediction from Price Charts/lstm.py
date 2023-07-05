import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.utils.rnn as rnn_utils
from torch.optim.lr_scheduler import StepLR
import copy
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR


'''
data = pd.read_csv('data_with_label.csv',sep=',',encoding='ANSI')
label = []
for i in range(data.shape[0]):
    if data['5_ret'][i] > 0:
        label.append(1)
    else:
        label.append(0)
data['label'] = label
data.to_csv('data.csv',sep=',',index=False)  
'''

data = pd.read_csv('data.csv',sep=',',encoding='gbk')

#将input和output使用标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = data[['code','date','open','high','low','close','volume','label']]
data_scaled = scaler.fit_transform(data.iloc[:, 2:7])   #eps_analyst,eps_outputs不需要进行标准化
data_scaled = pd.DataFrame(data_scaled)
df_label = data['label']
df_first= data[['code','date']]
data_new = pd.concat([df_first,data_scaled,df_label], axis=1)


#准备LSTM的input数据
windows_train  = [ ]
windows_test = []
pad_test = []
for company, data in data_new.groupby('code'):
    train_data = data.loc[data['date'] <= 20150830]
    window_train = train_data.iloc[:, 2:].values  
    num_samples = 300  #抽取这样的10个小序列   #100个数据太少了，学不到什么东西；1000可能太多了，数据冗余所以准确率不高；6000是61.88大概比300小一点点
    seq_length = 60
    for _ in range(num_samples):
        start_idx = np.random.randint(0, train_data.shape[0] - seq_length + 1)
        small_input = window_train [start_idx:start_idx + seq_length, :]
        windows_train.append(small_input) 
    
    test_data = data.loc[data['date'] >  20150830]
    window_test = test_data.iloc[:, 2:].values   
    if (window_test.shape[0] > 60+num_samples):
        for _ in range(num_samples):
            start_idx = np.random.randint(0, test_data.shape[0] - seq_length + 1)
            small_input = window_test [start_idx:start_idx + seq_length, :]
            windows_test.append(small_input)  
        
    '''
    for i in range(len(window_test) - seq_length):
        small_input_test = window_test [i:i + seq_length, :]
        windows_test.append(small_input_test)  #windows是一个三维变量，之后对每一页进行一一处理
    '''

windows_train = np.array(windows_train)
windows_test = np.array(windows_test)


# Define the finance dataset
class FinanceDataset(Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx,:, :5], self.data[idx,:, -1:]
    
    
# 将元组列表转换成二维数组
train_data = np.array(windows_train)
test_data = np.array(windows_test)

# Shuffle widows and pad_train, use random
idx = np.array(range(train_data.shape[0]))
np.random.shuffle(idx)
train_data = train_data[idx]


# Create the train and test datasets
all_dataset = FinanceDataset(train_data)
train_dataset, val_dataset = train_test_split(all_dataset, test_size=0.3, random_state=42)
test_dataset = FinanceDataset(test_data)


# Create the train and test data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Create the function to create LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        
        self.linear = nn.Linear(input_size, 32)
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear(x).view(-1, x.size(1), 32)
        outputs, (h, c) = self.lstm(x)
        out = self.fc(outputs[:,-1,:])
    
        return out


 
# Train the LSTM model
input_size = windows_train.shape[2] -1
hidden_size = 64
output_size = 2
num_layers = 3
num_epochs = 20
learning_rate = 0.001
weight_decay = 0
#随机抽取300个样本：train+test
#input_size = 32, hidden_size =64: 63.53, val最好是66.99
#input_size = 32, hidden_size =128: 62.46, val最好是66.99


#1层效果没有3层好，3层就够用了，5层太多了
# learning_rate = 0.0001准确率只有50%



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)  #自带softmax
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.5, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,betas = [0.9, 0.99])
#scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
#scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)


loss_list_train = []
loss_list_val = []
accuracy_list_train = []
accuracy_list_test = []

min_val_loss = 1e9  #保证第一次一定会更新
last_min_ind = -1
early_stopping_epoch = 2    #early stop

best_val_acc = 0
for epoch in range(num_epochs):
    total_loss = 0.0
    training_correct = 0
    val_correct = 0
    num_train = 0
    running_loss = 0.0
    val_current = 0
    num_val = 0
    
    print("Epoch", epoch + 1, "/", num_epochs, ":")
    model.train()
    with tqdm(train_loader) as tepoch:
        for i, (inputs, labels) in enumerate(tepoch):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(labels)
            one_hot_labels = F.one_hot(labels.long(), num_classes=2).to(torch.float)
            one_hot_labels = torch.squeeze(one_hot_labels, dim=2) 
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            train_outputs = model(inputs)
            #print(train_outputs.shape)
            #print(train_outputs)
            _val, pred = torch.max(train_outputs.data, 1)
            loss = criterion(train_outputs, one_hot_labels[:,-1,:])
            total_loss += loss.data
            loss.backward()
            optimizer.step()
            tepoch.set_description("losses {}".format(total_loss / (i + 1)))
            training_correct += torch.sum(pred == labels[:,-1,:].squeeze(1))
            num_train += len(labels)
        
         #逐步更新lr                    
        #scheduler.step()
        
    model.eval()   
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        one_hot_labels = F.one_hot(labels.long(), num_classes=2).to(torch.float)
        one_hot_labels = torch.squeeze(one_hot_labels, dim=2) 
        # calculate outputs by running images through the network
        val_outputs = model(inputs)
        # the class with the highest energy is what we choose as prediction
        _, pred = torch.max(val_outputs.data, 1)
        loss = criterion(val_outputs, one_hot_labels[:,-1,:])
        running_loss += loss.data
        val_correct += torch.sum(pred == labels[:,-1,:].squeeze(1))
        num_val += len(labels)
        
    print("Average training Loss is:", (total_loss / num_train).item(), 
          "Average test Loss is:", (running_loss / num_val).item(), 
        "Train Accuracy is:", (100 * training_correct / num_train).item(), "%",
        "Validation Accuracy is:", (100 * val_correct / num_val).item(), "%")

    if (100 * val_correct / num_val) > best_val_acc:
        best_val_acc = 100 * val_correct / num_val
        best_model = copy.deepcopy(model)
        print(best_val_acc)
        
        
    loss_list_train.append(round((total_loss / num_train).item(),3))
    loss_list_val.append(round((running_loss / num_val).item(),3))
    accuracy_list_train.append(round((training_correct / num_train).item(),2))
    accuracy_list_test.append(round((val_correct / num_val).item(),2))

    final_epoch = 0
    running_loss = running_loss / num_val
    #early stop
    if running_loss < min_val_loss:
        last_min_ind = epoch
        min_val_loss = running_loss    #检测每一次epoch之后val loss有没有变得更小
    elif epoch - last_min_ind >= early_stopping_epoch:
        final_epoch = epoch
        break
    
    
testing_correct = 0
test_loss = 0
best_model.eval()
num_test = 0
batch_test_acc = 0
TP,TN,FP,FN = 0,0,0,0
P,N = 0,0
sum = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():    
    for i, (images, labels) in enumerate(test_loader):   #最后在测试集上看效果
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = best_model(images)
        # the class with the highest energy is what we choose as prediction
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == labels[:,-1,:].squeeze(1))
        num_test += len(labels)
        TP += ((pred == labels[:,-1,:].squeeze(1))*(labels[:,-1,:].squeeze(1) == 1)).sum()
        TN += ((pred == labels[:,-1,:].squeeze(1))*(labels[:,-1,:].squeeze(1) == 0)).sum()
        FP += ((pred == labels[:,-1,:].squeeze(1))*(labels[:,-1,:].squeeze(1) == 0)).sum()
        FN += ((pred == labels[:,-1,:].squeeze(1))*(labels[:,-1,:].squeeze(1) == 1)).sum()
        sum += len(pred)
        P += TP+FP
        N += TN+FN
        if i % 100 == 0:
            fp=open('output.log','a+')
            print("{}: The batch accuracy is {}.".format(i, batch_test_acc/sum), file=fp)
            fp.close()
        fp=open('inference.log','a+')
        print("The test accuracy is {}.\n".format(batch_test_acc/sum), file=fp)
        print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)), file=fp)
        fp.close()
        print("The test accuracy is {}.\n".format(batch_test_acc/sum))
        print("TP is {}, TN is {}, FP is {}, FN is {}. Accuracy:{}, Precision:{}, Sensitivity:{}, Specificity:{}\n".format(TP, TN, FP, FN, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)))


print("Test Accuracy is:", (100 * testing_correct / num_test).item(), "%")
torch.save(best_model.state_dict(), '_model_2.pth')

#画图
""" Plot loss and accuracy curve """
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
if (final_epoch != 0):
    maxEpoch = final_epoch+1
else:
    maxEpoch = epoch +1 
    
maxLoss = max(loss_list_train) 
minLoss = max(0, min(loss_list_train))
plt.plot(range(1, 1 + maxEpoch), loss_list_train,'-s', label='train loss')
plt.plot(range(1, 1 + maxEpoch), loss_list_val,'-s', label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(range(0, maxEpoch + 1, 2))
plt.axis([0, maxEpoch, 0.007, 0.012])
plt.savefig('LSTM_loss_2.png', dpi=300)


fig = plt.figure()
plt.plot(range(1, 1 + maxEpoch), accuracy_list_train, '-s', label='train accuracy')
plt.plot(range(1, 1 + maxEpoch), accuracy_list_test, '-s', label='Valiadtion accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(range(0, maxEpoch + 1, 2))
plt.axis([0, maxEpoch, 0.2 ,1])
plt.legend()
plt.savefig('LSTM_acc_2.png', dpi=300)
