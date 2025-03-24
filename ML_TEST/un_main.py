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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
parser.add_argument('--dataset_name', type=str, default='VGGFACE')
parser.add_argument('--num_class',type =int, default=100)
parser.add_argument('--class_unlearned', default=[0,1,2],nargs='+',type =int)
parser.add_argument('--lr',type =float, default=0.00005)
parser.add_argument('--unlearn_epochs',type =int, default= 20 )

parser.add_argument('--batch_size', default=64,nargs='+',type =int)
parser.add_argument('--model_name',type =str, default='ResNet')
parser.add_argument('--epochs',type =int, default=50)
parser.add_argument('--save_model_file',type =str, default='./ML_TEST/un_model_file/')
parser.add_argument('--save_data_file',type =str, default='./ML_TEST/un_info_file/')
params=parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
train_dataset,test_dataset = get_data.return_data(params)
test_datalaoder = DataLoader(dataset = test_dataset,batch_size=params.batch_size,shuffle=False)
print(params.class_unlearned)
def is_class(target):
    return target in params.class_unlearned
def is_classr(target):
    return target not in params.class_unlearned

#test data
from torch.utils.data import Subset,DataLoader
class0_dataset_train = Subset(train_dataset, [i for i, (_, sample_label) in enumerate(train_dataset) if is_class(sample_label)])
train_loader0 = DataLoader(dataset=class0_dataset_train, batch_size=64, shuffle=False) #用目标数据进行遗忘

class0_dataset_test = Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if is_class(sample_label)])
test_loader0 = DataLoader(dataset=class0_dataset_test, batch_size=64, shuffle=False)
classr_dataset_test = Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if is_classr(sample_label)])
test_loader_r = DataLoader(dataset=classr_dataset_test, batch_size=64, shuffle=False)

#软硬样本划分
npresentations = params.epochs
unlearned_per_presentation = {}

#文件加载
if params.dataset_name == "VGGFACE":
    a = torch.load("./second/ML_TEST/data_info_file/VGGFACE_DenseNet_epoch20.pth")
    unlearn_model_dict = torch.load('./second/ML_TEST/model_file/VGGFACE_DenseNet20.pth',weights_only=True)
if params.dataset_name == "FMNIST":
    a = torch.load("./ML_TEST/data_info_file/FMNIST_Lenet5_epoch50.pth")
    unlearn_model_dict = torch.load('./ML_TEST/model_file/FMNIST_Lenet550.pth',weights_only=True)
if params.dataset_name == "PLANT":
    a = torch.load("./ML_TEST/data_info_file/PLANT_MobileNetV2_epoch20.pth")
    unlearn_model_dict = torch.load('./ML_TEST/model_file/PLANT_MobileNetV220.pth',weights_only=True)
if params.dataset_name == "CIFAR":
    a = torch.load("./ML_TEST/data_info_file/CIFAR_ResNet_epoch50.pth")
    unlearn_model_dict = torch.load('./ML_TEST/model_file/CIFAR_ResNet50.pth',weights_only=True)
if params.dataset_name == "SVHN":
    a = torch.load("./ML_TEST/data_info_file/SVHN_VGG11_epoch20.pth")
    unlearn_model_dict = torch.load('./ML_TEST/model_file/SVHN_VGG1120.pth',weights_only=True)

for example_id, example_stats in a.items():
    presentation_acc = np.array(example_stats[:npresentations])
    transitions = presentation_acc[1:] - presentation_acc[:-1]
    #为负一则会被遗忘
    if len(np.where(transitions == -1)[0]) > 0:
        unlearned_per_presentation[example_id] = np.where(transitions == -1)[0] + 2 #调整索引顺序
    else:
        unlearned_per_presentation[example_id] = []
        #unlearning_per_presen 这个变量key 为所有样本的id 值为样本出现遗忘的轮数
        #presentation_acc = np.array(a)
        #unlearned_per_presentation_all.append(unlearned_per_presentation)
# 对汇总进行排序
example_original_order = []  # 样本的原始顺序
example_stats = []  # 每个样本的总遗忘次数
for example_id in unlearned_per_presentation.keys():
    example_original_order.append(example_id) #id
    example_stats.append(0) #遗忘次数
    example_stats[-1] += len(unlearned_per_presentation[example_id])
########################################################################################################
soft_sample_0 = Subset(train_dataset, [value for i,value in enumerate(example_original_order) if example_stats[i]==0 and train_dataset.targets[value] in params.class_unlearned])
hard_sample_0= Subset(train_dataset, [value for i,value in enumerate(example_original_order) if example_stats[i]!=0 and train_dataset.targets[value] in params.class_unlearned])
soft_sample_r = Subset(train_dataset, [value for i,value in enumerate(example_original_order) if example_stats[i]==0 and train_dataset.targets[value] not in params.class_unlearned])
hard_sample_r = Subset(train_dataset, [value for i,value in enumerate(example_original_order) if example_stats[i]!=0 and train_dataset.targets[value] not in params.class_unlearned])
print("软硬样本数量之和",len(soft_sample_0),len(soft_sample_r),len(hard_sample_0),len(hard_sample_r))

soft_sample_0_loder = DataLoader(dataset=soft_sample_0, batch_size=64, shuffle=False)
hard_sample_0_loder = DataLoader(dataset=hard_sample_0, batch_size=64, shuffle=False)
soft_sample_r_loder = DataLoader(dataset=soft_sample_r, batch_size=64, shuffle=False)
hard_sample_r_loder = DataLoader(dataset=hard_sample_r, batch_size=64, shuffle=False)

unlearn_model = get_model.return_model(params)
unlearn_model.load_state_dict(unlearn_model_dict)
unlearn_model.eval()

optimizer_un = optim.SGD(unlearn_model.parameters(), lr=params.lr,momentum=0.9)
print(optimizer_un)
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)
all_acc1 = []
all_acc2 = []
all_acc3 = []
all_acc4 = []
all_acc5 = []
all_acc6 = []
acc1 = return_acc(unlearn_model, soft_sample_0_loder)
acc2 = return_acc(unlearn_model, hard_sample_0_loder)
acc3 = return_acc(unlearn_model, soft_sample_r_loder)
acc4 = return_acc(unlearn_model, hard_sample_r_loder)
acc5 = return_acc(unlearn_model, test_loader0)
acc6 = return_acc(unlearn_model, test_loader_r)
all_acc1.append(acc1)
all_acc2.append(acc2)
all_acc3.append(acc3)
all_acc4.append(acc4)
all_acc5.append(acc5)
all_acc6.append(acc6)
print("首次",acc1,acc2,acc3,acc4,acc5,acc6)
for epoch in range(params.unlearn_epochs):
    unlearn_model.train()
    unlearn_model.to(device)
    for data, target in train_loader0:
        data, target = data.to(device), target.to(device)
        optimizer_un.zero_grad()
        output = unlearn_model(data)
        random_tensor = torch.tensor([random.choice([x for x in range(params.num_class) if x != target[index]]) for index in range(len(target))]).cuda()
        loss = torch.mean(criterion(output, random_tensor))
        loss.backward()
        optimizer_un.step()
    unlearn_model.eval()
    acc1 = return_acc(unlearn_model, soft_sample_0_loder)
    acc2 = return_acc(unlearn_model, hard_sample_0_loder)
    acc3 = return_acc(unlearn_model, soft_sample_r_loder)
    acc4 = return_acc(unlearn_model, hard_sample_r_loder)
    acc5 = return_acc(unlearn_model, test_loader0)
    acc6 = return_acc(unlearn_model, test_loader_r)
    print("本轮",acc1,acc2,acc3,acc4,acc5,acc6)
    all_acc1.append(acc1)
    all_acc2.append(acc2)
    all_acc3.append(acc3)
    all_acc4.append(acc4)
    all_acc5.append(acc5)
    all_acc6.append(acc6)
A = [all_acc1,all_acc2,all_acc3,all_acc4,all_acc5,all_acc6]
torch.save(unlearn_model.state_dict(),params.save_model_file+params.dataset_name+params.model_name+"_"+'.pth')
torch.save(A,params.save_data_file+params.dataset_name+'.pth')