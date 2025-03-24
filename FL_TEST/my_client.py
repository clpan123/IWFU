import copy

from torch.utils.data import Subset,DataLoader
import get_data,get_model
import torch
import torch.nn as nn
from utils import UnlearningLoss,AdaHessian
import torch.nn.functional as F
import os
import gc
import numpy as np
import time
import random
import gc
from utils import return_acc

class FLclient():
    def __init__(self,params,train_dataset):
        self.params = params
        self.parameters_ = None
        self.train_loader = None
        self.train_dataset = train_dataset
        self.train_loader = None
        self.unlearn_dataloder = None
        self.recovery_dataloder = None
        self.fisher_ewc = None
        self.original_model = None
        self.sample_index = None
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.__init__(reduce=False)
        self.example_stats = {}
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        self.near_label = []
        self.unlearn_dataset = None
        self.recovery_dataset = None

    def recieve_model(self,parameters):
        self.parameters_ = parameters     

    def train_A(self):######################################################################  FL
        model = get_model.return_model(self.params)
        model.load_state_dict(self.parameters_)
        model.train()
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr,momentum=0.9)
        for epoch in range(self.params.local_epochs):
            seed = 0
            np.random.seed(seed)
            seed+=1
            model.train()
            model.to(self.device)
            trainset_permutation_inds = np.random.permutation(np.arange(len(self.train_dataset.targets)))
            for batch_idx, batch_start_ind in enumerate(range(0, len(self.train_dataset), self.params.batch_size)):
                batch_inds = trainset_permutation_inds[batch_start_ind:batch_start_ind + self.params.batch_size]
                transformed_trainset = []
                for ind in batch_inds:
                    transformed_trainset.append(self.train_dataset[ind][0])
                inputs = torch.stack(transformed_trainset)
                targets = torch.LongTensor(np.array(self.train_dataset.targets)[batch_inds].tolist())
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        return model.state_dict(),len(self.train_dataset)
    
    def train_B(self,):
        ############################################################################
        # UN training
        model = get_model.return_model(self.params)
        model.load_state_dict(self.parameters_)
        model = model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.unlearn_lr,momentum=0.9)
        if self.unlearn_dataset == None:
            self.unlearn_dataset = Subset(self.train_dataset, [i for i, (_, sample_label) in enumerate(self.train_dataset) if sample_label in self.params.class_unlearned])
        if self.unlearn_dataloder ==None:
            self.unlearn_dataloder = DataLoader(dataset=self.unlearn_dataset, batch_size=64,shuffle=True)
        for epoch in range(self.params.local_epochs):
            torch.cuda.empty_cache()  # 释放显存
            for i, (train_batch, labels_batch) in enumerate(self.unlearn_dataloder):
                model.train()
                optimizer.zero_grad()
                train_batch = train_batch.cuda()
                labels_batch = labels_batch.cuda()
                output_batch= model(train_batch)

                if self.params.mode.startswith("nearst"):
                    sfx_output_batch = torch.nn.functional.softmax(output_batch,dim=1)
                    mask = torch.ones_like(output_batch, dtype=torch.bool)
                    mask[torch.arange(output_batch.shape[0]), labels_batch] = False
                    max_values, nearst_label = torch.max(sfx_output_batch * mask, dim=1)
                    if self.params.mode == "nearst_label_reweight":
                        unlearn_loss = UnlearningLoss(nn.CrossEntropyLoss())
                        localloss = unlearn_loss(output_batch, nearst_label,[sfx_output_batch[i][labels_batch[i]]- max_values[i] for i in range(len(labels_batch))])
                    else:
                        localloss = nn.CrossEntropyLoss()(output_batch, nearst_label)
                if self.params.mode == "random_label":
                    localloss = nn.CrossEntropyLoss()(output_batch, torch.tensor([random.choice([x for x in range(self.params.num_class) if x != labels_batch[index]]) for index in range(len(labels_batch))]).cuda())       
                torch.mean(localloss).backward()
                optimizer.step()
                del train_batch, labels_batch
        dict_ = model.state_dict()
        del model
        return dict_,len(self.unlearn_dataset)
    
    def train_C(self,):
        model = get_model.return_model(self.params)
        model.load_state_dict(self.parameters_)
        model = model.cuda()

        ZL_model = copy.deepcopy(model)
        distillation_criterion = nn.KLDivLoss(reduction='batchmean')
        ZL_model.eval()
        ZL_model = ZL_model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.params.unlearn_lr,momentum=0.9)
        if self.recovery_dataset == None:
            self.recovery_dataset = Subset(self.train_dataset, [i for i, (_, sample_label) in enumerate(self.train_dataset) if sample_label not in self.params.class_unlearned])
        if self.recovery_dataloder ==None:
            self.recovery_dataloder = DataLoader(dataset=get_data.classify_one(self.train_dataset, [i for i in range(self.params.num_class) if i not in self.params.class_unlearned]), batch_size=64,shuffle=True)

        for epoch in range(self.params.local_epochs):
            for i, (train_batch, labels_batch) in enumerate(self.recovery_dataloder):
                model.train()
                optimizer.zero_grad()
                train_batch = train_batch.cuda()
                labels_batch = labels_batch.cuda()
                output_batch= model(train_batch)
                if self.params.ZL>0:
                    with torch.no_grad():
                        output_batch_teacher = ZL_model(train_batch)
                    dis_loss = distillation_criterion(nn.functional.log_softmax(output_batch[:,:len(self.params.class_unlearned)]/self.params.ZL, dim=1),
        nn.functional.softmax(output_batch_teacher[:,:len(self.params.class_unlearned)]/self.params.ZL, dim=1))
                    localloss = torch.mean(nn.CrossEntropyLoss()(output_batch, labels_batch)) + self.params.ratio* dis_loss
                else:
                    localloss = torch.mean(self.criterion(output_batch, labels_batch))
                localloss.backward()
                optimizer.step()
        return model.state_dict(),len(self.recovery_dataset)