import collections
import copy
import torch
import get_model

class FLserver():
    def __init__(self,params):
        self.model = get_model.return_model(params)
        self.params = params

    def rec_model(self,low_model):
        #受到模型
        for name, param in low_model.state_dict().items():
            self.model.state_dict()[name].copy_(param.clone())
    def agg(self,model_list,weight_list):
        # models 是个列表.
        worker_state_dict = model_list
        all_num = sum(weight_list)
        #拿到网络的key
        weight_keys = list(worker_state_dict[0].keys())
        #创建有序字典，保存模型更新的权重
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(model_list)):
                key_sum = key_sum + weight_list[i]*worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / all_num
        self.model.load_state_dict(fed_state_dict)