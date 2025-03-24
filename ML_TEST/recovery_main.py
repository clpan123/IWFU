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
import get_data,get_model
import argparse
from utils import return_acc
from torch.utils.data import DataLoader
import copy
import warnings
warnings.filterwarnings("ignore")

from torchvision import datasets, transforms


if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

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
parser.add_argument('--dataset_name', type=str, default='VGGFACE')
parser.add_argument('--num_class',type =int, default=100)
parser.add_argument('--class_unlearned', default=[0,1,2],nargs='+',type =int)
parser.add_argument('--lr',type =float, default=0.01)
parser.add_argument('--unlearn_epochs',type =int, default=20)
parser.add_argument('--batch_size', default=64,type =int)
parser.add_argument('--model_name',type =str, default='ResNet')
parser.add_argument('--epochs',type =int, default=50)
parser.add_argument('--save_model_file',type =str, default='./ML_TEST/un_model_file/')
parser.add_argument('--save_data_file',type =str, default='./ML_TEST/recovery_info_file/')
params=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
train_dataset,test_dataset = get_data.return_data(params)
test_datalaoder = DataLoader(dataset = test_dataset,batch_size=params.batch_size,shuffle=False)
print(params)
def is_class(target):
    return target in params.class_unlearned
def is_classr(target):
    return target not in params.class_unlearned

from torch.utils.data import Subset,DataLoader
classr_dataset_train = Subset(train_dataset, [i for i, (_, sample_label) in enumerate(train_dataset) if is_classr(sample_label)])
train_loaderr = DataLoader(dataset=classr_dataset_train, batch_size=64, shuffle=True) #这个可以shuffle

class0_dataset_test = Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if is_class(sample_label)])
test_loader0 = DataLoader(dataset=class0_dataset_test, batch_size=64, shuffle=False)
classr_dataset_test = Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if is_classr(sample_label)])
test_loader_r = DataLoader(dataset=classr_dataset_test, batch_size=64, shuffle=False)

#加载文件
if params.dataset_name == "VGGFACE":
    unlearn_model_dict = torch.load('./second/ML_TEST/un_model_file/VGGFACEDenseNet_.pth')
if params.dataset_name == "FMNIST":
    unlearn_model_dict = torch.load('./ML_TEST/un_model_file/FMNIST.pth')
if params.dataset_name == "PLANT":
    unlearn_model_dict = torch.load('./ML_TEST/un_model_file/PLANTMobileNetV2_.pth')
if params.dataset_name == "CIFAR":
    unlearn_model_dict = torch.load('./ML_TEST/un_model_file/CIFAR.pth')
if params.dataset_name == "SVHN":
    unlearn_model_dict = torch.load('./ML_TEST/un_model_file/SVHNVGG11_.pth')

unlearn_model = get_model.return_model(params)
unlearn_model.load_state_dict(unlearn_model_dict)

optimizer_un = optim.SGD(unlearn_model.parameters(), lr=params.lr,momentum=0.9)
print(optimizer_un)
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce='none')
all_acc5 = []
all_acc6 = []
acc5 = return_acc(unlearn_model, test_loader0)
acc6 = return_acc(unlearn_model, test_loader_r)
all_acc5.append(acc5)
all_acc6.append(acc6)
print("初始精度",acc5,acc6)

unlearn_model.train()
unlearn_model.to(device)
for epoch in range(params.unlearn_epochs):
    unlearn_model.train()
    for data, target in train_loaderr:
        data, target = data.to(device), target.to(device)
        # print(target)
        optimizer_un.zero_grad()
        output = unlearn_model(data)
        loss = criterion(output, target)
        loss = loss.mean()
        # print(loss)
        loss.backward()
        optimizer_un.step()
    acc5 = return_acc(unlearn_model, test_loader0)
    acc6 = return_acc(unlearn_model, test_loader_r)
    print("精度为",acc5,acc6)
    all_acc5.append(acc5)
    all_acc6.append(acc6)
#模型精度存储
torch.save([all_acc5,all_acc6],params.save_data_file+params.dataset_name+'.pth')