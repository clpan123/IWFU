import torch
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def return_acc(model,dataloader):
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        acc = round(correct/total,4)
    # print("本次测试精度为:",acc)
    return acc

#自定义子数据集合
from torch.utils.data import DataLoader,Dataset
# from torch. import Dataset
class CustomDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        # self.targets = dataset.targets # 保留targets属性
        self.classes = dataset.classes # 保留classes属性
        self.targets = self.get_targets() # 保留targets属性
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        return x, y
    
    def get_targets(self):
        targets = []
        for i in range(len(self.indices)):
            targets.append(self.dataset.targets[self.indices[i]])
        return torch.tensor(targets)

import math
import torch.nn as nn
class UnlearningLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(UnlearningLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'
    def forward(self, pred, true,distance, auto_iou=0.5):
        loss = self.loss_fcn(pred, true) #IOU
        if auto_iou < 0.2:
            auto_iou = 0.2
        # b1 = true <= auto_iou - 0.1
        # a1 = 1.0
        # b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        # a2 = math.exp(1.0 - auto_iou)
        # b3 = true >= auto_iou
        # a3 = torch.exp(-(true - 1.0))
        a1 = torch.exp(torch.tensor(distance))
        modulating_weight = a1.cuda()
        # print(modulating_weight)
        # print(len(modulating_weight))
        # print(distance)
        # print(len(a1))
        breakpoint
        # print(len(loss))
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss