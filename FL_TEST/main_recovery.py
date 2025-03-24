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
parser.add_argument('--dataset_name', type=str, default='MNIST')
parser.add_argument('--num_class',type =int, default=38)
parser.add_argument('--class_unlearned', default=[0,1,2],nargs='+',type =int)
parser.add_argument('--batch_size', type =int, default=64)
parser.add_argument('--model_name',type =str, default='Lenet5')
parser.add_argument('--global_epochs',type =int, default=10)
parser.add_argument('--local_epochs',type =int, default=2)
parser.add_argument('--lr',type =float, default=0.01)
parser.add_argument('--unlearn_lr',type =float, default=0.01)
parser.add_argument('--client_number',type =int, default=10)
parser.add_argument('--mode',type =str, default="normal_recovery")
parser.add_argument('--ZL',type =float, default=0.1) 
parser.add_argument('--ratio',type =float, default=1)
parser.add_argument('--alpha_FL_dis',type =float, default=1)
parser.add_argument('--save_model_file',type =str, default='./FL_TEST/FL_un_model_file/')
parser.add_argument('--save_data_file',type =str, default='./FL_TEST/acc_recovery/')
params=parser.parse_args()
print(params)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
############################################################################################################
#####################################数据重新加载
train_dataset,test_dataset = get_data.return_data(params)
Un_test_dataloder = DataLoader(dataset=Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if sample_label in params.class_unlearned]), batch_size=128, shuffle=False, num_workers=0, drop_last=False)
Pre_test_dataloder = DataLoader(dataset=Subset(test_dataset, [i for i, (_, sample_label) in enumerate(test_dataset) if sample_label not in params.class_unlearned]), batch_size=128, shuffle=False, num_workers=0, drop_last=False)
FL_dataset = get_data.dirichlet_split_noniid(np.array(train_dataset.targets),params.client_number,alpha=params.alpha_FL_dis)
####################################################################
##################### 服务器 加载已经训练好的模型 客户端构建样本池##########################
my_server = My_server.FLserver(params)
if params.client_number == 10 and params.alpha_FL_dis == 1.0:
    if params.dataset_name == "FMNIST":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/FMNIST_Lenet5_10_clientnum1010_FLalpha1.0.pth')
    if params.dataset_name == "CIFAR":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/CIFAR_ResNet_10_clientnum1010_FLalpha1.0.pth')
    if params.dataset_name == "VGGFACE":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/VGGFACE_DenseNet_10_clientnum1010_FLalpha1.0.pth')
    if params.dataset_name == "SVHN":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/SVHN_VGG11_10_clientnum1010_FLalpha1.0.pth')

if params.client_number == 20 and params.alpha_FL_dis == 1.0:
    if params.dataset_name == "FMNIST":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/FMNIST_Lenet5_10_clientnum2010_FLalpha1.0.pth')
    if params.dataset_name == "CIFAR":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/CIFAR_ResNet_10_clientnum2010_FLalpha1.0.pth')
    if params.dataset_name == "VGGFACE":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/VGGFACE_DenseNet_10_clientnum2010_FLalpha1.0.pth')
    if params.dataset_name == "SVHN":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/SVHN_VGG11_10_clientnum2010_FLalpha1.0.pth')

if params.client_number == 10 and params.alpha_FL_dis == 2.0:
    if params.dataset_name == "FMNIST":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/FMNIST_Lenet5_10_clientnum1010_FLalpha2.0.pth')
    if params.dataset_name == "CIFAR":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/CIFAR_ResNet_10_clientnum1010_FLalpha2.0.pth')
    if params.dataset_name == "VGGFACE":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/VGGFACE_DenseNet_10_clientnum1010_FLalpha2.0.pth')
    if params.dataset_name == "SVHN":
        unlearn_model_dict = torch.load('./FL_TEST/FL_un_model_file/SVHN_VGG11_10_clientnum1010_FLalpha2.0.pth')



my_server.model.load_state_dict(unlearn_model_dict)
myclient= []
# data_ = [a, b, c, d, e ,f ,g ,h ,i, j]
for i in range(params.client_number):
    cle = my_client.FLclient(params,CustomDataset(train_dataset,FL_dataset[i]))
    myclient.append(cle)

##################################################################
######################  FL local unlearning train #################################
def FL_local_train():
    for i in myclient:
        i.recieve_model(my_server.model.state_dict())
    choose_client = random.sample(myclient,10)
    avg_model_parameters = []
    all_data_len = []
    for i in choose_client:
        client_update_model_parameter,data_len = i.train_C()
        all_data_len.append(data_len)
        avg_model_parameters.append(client_update_model_parameter)
    my_server.agg(avg_model_parameters,all_data_len)
###################################################################
#######################  FL global train #################################
maxacc = 0
global_train_acc=[]
H_list = []
acc1_list = []
acc2_list = []
acc1_fir = return_acc(my_server.model, Un_test_dataloder)
acc2_fir = return_acc(my_server.model, Pre_test_dataloder)
acc1_list.append(acc1_fir)
acc2_list.append(acc2_fir)
H_list.append(0)
print("首次精度",acc1_fir,acc2_fir)
start_time = time.time()
for epoch in range(params.global_epochs):
     FL_local_train()
     acc1 = return_acc(my_server.model, Un_test_dataloder)
     acc2 = return_acc(my_server.model, Pre_test_dataloder)
     acc1_list.append(acc1)
     acc2_list.append(acc2)
     H_list.append(2*(acc1_fir-acc1)*acc2/((acc1_fir-acc1)+acc2))
     print("本次FL全局模型训练精度为", acc1,acc2)
end_time = time.time()
run_time = end_time - start_time
print("运行时间为",run_time)
last_acc = [acc1_list,acc2_list,run_time]
nick_name = params.dataset_name + '_'+ params.mode +'_'+str(params.ZL )+ "_epoch"+ str(params.global_epochs) + "ratio"+str(params.ratio)+ '_clientnum'+str(params.client_number)+'_FLalpha'+str(params.alpha_FL_dis)+".pth"
# torch.save(last_acc,params.save_data_file + nick_name)
# torch.save(my_server.model.state_dict(),'FL_TEST/compara_acc/DOWN/CIFAR_finish_FOR_DOWNKL.pt')