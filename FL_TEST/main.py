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
from torch.utils.data import DataLoader,Subset
from utils import CustomDataset
import My_server,my_client
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
parser.add_argument('--dataset_name', type=str, default='FMNIST')
parser.add_argument('--num_class',type =int, default=10)
parser.add_argument('--class_unlearned', default=[5],nargs='+',type =int)
parser.add_argument('--batch_size', type =int, default=64)
parser.add_argument('--model_name',type =str, default='Lenet5')
parser.add_argument('--global_epochs',type =int, default=2)
parser.add_argument('--local_epochs',type =int, default=2)
parser.add_argument('--lr',type =float, default=0.01)
parser.add_argument('--client_number',type =int, default=10)
parser.add_argument('--alpha_FL_dis',type =float, default=1)
parser.add_argument('--save_model_file',type =str, default='./FL_TEST/FL_model_file/')
parser.add_argument('--save_data_file',type =str, default='./FL_TEST/data_info_file/')
params=parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
############################################################################################################
#####################################数据 模型
train_dataset,test_dataset = get_data.return_data(params)
test_dataloder = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
FL_dataset = get_data.dirichlet_split_noniid(np.array(train_dataset.targets),params.client_number,alpha=params.alpha_FL_dis)
####################################################################
##################### 服务器 客户端 初始化 ##########################
my_server = My_server.FLserver(params)
myclient= []
# data_ = [a, b, c, d, e ,f ,g ,h ,i, j]
for i in range(params.client_number):
    cle = my_client.FLclient(params,CustomDataset(train_dataset,FL_dataset[i]))
    myclient.append(cle)

##################################################################
######################   FL local train #################################
def FL_local_train():
    for i in myclient:
        i.recieve_model(my_server.model.state_dict())
    choose_client = random.sample(myclient,int(0.6*params.client_number))
    avg_model_parameters = []
    all_data_len = []
    for i in choose_client:
        client_update_model_parameter,data_len = i.train_A()
        all_data_len.append(data_len)
        avg_model_parameters.append(client_update_model_parameter)
    my_server.agg(avg_model_parameters,all_data_len)
###################################################################  
#######################  FL global train #################################
maxacc = 0
global_train_acc=[]
H_list = []
for epoch in range(params.global_epochs):
	FL_local_train()
	acc = return_acc(my_server.model, test_dataloder)
	print("本次FL全局模型训练精度为", acc)
	global_train_acc.append(acc)
nick_name = params.dataset_name + '_' + params.model_name+ "_epoch"+ str(params.global_epochs) + ".pth"
torch.save(my_server.model.state_dict(),params.save_model_file+params.dataset_name + '_' + params.model_name +'_' + str(params.global_epochs)+'_clientnum'+str(params.client_number)+str(params.global_epochs)+'_FLalpha'+str(params.alpha_FL_dis) +".pth")