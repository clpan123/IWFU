import argparse
import numpy as np
import numpy.random as npr
import time
import os
import sys
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os
import get_data,get_model
import argparse
from utils import return_acc
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(0)

########### 参数
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--dataset_name', type=str, default='CIFAR100')
parser.add_argument('--num_class',type =int, default=100)
parser.add_argument('--class_unlearned', default=[0,1,2],nargs='+',type =int)
parser.add_argument('--batch_size', default=64,type =int)
parser.add_argument('--model_name',type =str, default='MobileNetV2')
parser.add_argument('--epochs',type =int, default=20)
parser.add_argument('--lr',type =float, default=0.01)

parser.add_argument('--save_model_file',type =str, default='./ML_TEST/model_file/')
parser.add_argument('--save_data_file',type =str, default='./ML_TEST/data_info_file/')
params=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

train_dataset,test_dataset = get_data.return_data(params)
test_datalaoder = DataLoader(dataset = test_dataset,batch_size=params.batch_size,shuffle=False)

train_indx = np.array(range(len(train_dataset.targets)))
print(type(train_indx))
print(train_dataset.data[0].shape)
train_dataset.data = train_dataset.data[train_indx, :, :]
train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()

my_model  = get_model.return_model(params)
optimizer = optim.SGD(my_model.parameters(), lr=params.lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)
example_stats = {}
print(params)
#############################################  train
for epoch in range(params.epochs):
    correct = 0
    total = 0
    seed = 0
    np.random.seed(seed)
    seed+=1
    my_model.train()
    my_model.to(device)
    trainset_permutation_inds = npr.permutation(np.arange(len(train_dataset.targets)))
    for batch_idx, batch_start_ind in enumerate(range(0, len(train_dataset), params.batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_ind:batch_start_ind + params.batch_size]
        #得到这个batch的数据的索引（随机选取数据的索引）
        transformed_trainset = [] #根据随机索引拿出来数据
        for ind in batch_inds:
            transformed_trainset.append(train_dataset[ind][0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(np.array(train_dataset.targets)[batch_inds].tolist())
        #拿到了数据和标签
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = my_model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        # torch max返回两个值 outputs ： value 和 索引  用data是因为可以避免求梯度
        #更新统计信息
        acc = predicted == targets
        for j, index in enumerate(batch_inds):
            #遍历每个值观察内部样本：（enumrate 先给索引再给值）
            index_in_original_dataset = train_indx[index]
            #拿到原始数据
            #以下是计算margain 距离
            #计算精度
            #信息存储
            index_stats = example_stats.get(index_in_original_dataset,[])
            index_stats.append(acc[j].sum().item())  #当前是否是正确的
            example_stats[index_in_original_dataset] = index_stats
        #更新损失 回传网络 优化更新
        loss = loss.mean()#求平均损失
        loss.backward()
        optimizer.step()
        #字典值，将数值赋给index_stats
        # index_stats = example_stats.get('train', [[], []])
        # index_stats[1].append(100. * correct.item() / float(total))
        # example_stats['train'] = index_stats
    acc = return_acc(my_model, test_datalaoder)
    print("精度为",acc)
nick_name = params.dataset_name + '_' + params.model_name+ "_epoch"+ str(params.epochs) + ".pth"
torch.save(example_stats,params.save_data_file + nick_name)
torch.save(my_model.state_dict(),params.save_model_file+params.dataset_name + '_' + params.model_name + str(params.epochs) +".pth")
#字典存储 和 模型存储