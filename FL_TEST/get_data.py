
from torchvision import datasets, transforms
import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
from PIL import Image

class MLGB(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(MLGB, self).__init__()
        self.data = DataArray
        self.label = LabelArray
        self.targets = LabelArray
        self.classes = [i for i in range(38)]

    def __getitem__(self, index):
        resize = (224,224)
        img = Image.fromarray(self.data[index])
        im_trans = transforms.Compose([
            transforms.Resize(int(max(resize))),
            transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.4118, 0.4903, 0.4664], [0.1871, 0.1432, 0.1705])
        ])
        return im_trans(img), torch.tensor(self.label[index], dtype=torch.long)
    def __len__(self):
        return self.label.shape[0]

class MLGB_test(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(MLGB_test, self).__init__()
        self.data = DataArray
        self.label = LabelArray
        self.targets = LabelArray
        self.classes = [i for i in range(38)]
    def __getitem__(self, index):
        resize = (224,224)
        img = Image.fromarray(self.data[index])
        im_trans = transforms.Compose([
            transforms.Resize(int(max(resize))),
            transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.4118, 0.4903, 0.4664], [0.1871, 0.1432, 0.1705])
        ])
        return im_trans(img), torch.tensor(self.label[index], dtype=torch.long)
    def __len__(self):
        return len(self.label)

class SVHN_traindata(Dataset):
    def __init__(self):
        super(SVHN_traindata, self).__init__()
        self.trainset = datasets.SVHN(
            root='/hy-tmp/',
            split='train',
            download=True,
            transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
        )
        self.data = [self.trainset[i][0] for i in range(len(self.trainset))]
        self.label = [self.trainset[i][1] for i in range(len(self.trainset))]
        # print(self.label)
        # print(self.label)
        self.targets = self.label
        self.classes = [i for i in range(10)]
    def __getitem__(self, index):
        return self.trainset[index]
    def __len__(self):
        return len(self.label)

class SVHN_testdata(Dataset):
    def __init__(self):
        super(SVHN_testdata, self).__init__()
        self.testset = datasets.SVHN(
            root='/hy-tmp/',
            split='test',
            download=True,
            transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
        )
        self.data = [self.testset[i][0] for i in range(len(self.testset))]
        self.label = [self.testset[i][1] for i in range(len(self.testset))]
        # print(self.label)
        # print(self.label)
        self.targets = self.label
        self.classes = [i for i in range(10)]
    def __getitem__(self, index):
        return self.testset[index]
    def __len__(self):
        return len(self.label)
    


class Lacuna100(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(Lacuna100, self).__init__()
        self.data = DataArray
        self.label = LabelArray
        self.targets = LabelArray
        self.classes = [i for i in range(100)]

    def __getitem__(self, index):
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.3731, 0.4233, 0.5309], [0.2115, 0.2214, 0.2516])
            ]
        )
        return im_trans(self.data[index]), torch.tensor(self.label[index], dtype=torch.long)

    def __len__(self):
        return self.label.shape[0]

class Lacuna100_test(Dataset):
    def __init__(self, DataArray, LabelArray):
        super(Lacuna100_test, self).__init__()
        self.data = DataArray
        self.label = LabelArray
        self.targets = LabelArray
        self.classes = [i for i in range(100)]

    def __getitem__(self, index):
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.3731, 0.4233, 0.5309], [0.2115, 0.2214, 0.2516])
            ]
        )
        return im_trans(self.data[index]), torch.tensor(self.label[index], dtype=torch.long)
    
    def __len__(self):
        return self.label.shape[0]
    
    

def return_data(params):
    dataset = params.dataset_name
    if dataset == "MNIST":
        all_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])
        trainset = datasets.MNIST(root='/hy-tmp/', train=True, download=True, transform=all_transforms)
        testset = datasets.MNIST(root='/hy-tmp/', train=False, download=True, transform=all_transforms)
    if dataset == "FMNIST":
        all_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])
        trainset = datasets.FashionMNIST(root='/hy-tmp/', train=True, download=True, transform=all_transforms)
        testset = datasets.FashionMNIST(root='/hy-tmp/', train=False, download=True, transform=all_transforms)
    if dataset == "CIFAR":
        train_all_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2610))])
        test_all_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2610))])
        trainset = datasets.CIFAR10(root='/hy-tmp/', train=True, download=True, transform=train_all_transforms)
        testset = datasets.CIFAR10(root='/hy-tmp/', train=False, download=True, transform=test_all_transforms)
    if dataset == "SVHN":
        trainset = SVHN_traindata()
        testset = SVHN_testdata()

    if dataset == "VGGFACE":
        #加载vggface数据集合
        path = r'/hy-tmp/v100/v100/'
        pathlist = map(lambda x: '/'.join([path, x]), os.listdir(path))
        namedict = {}
        data, label = [], []
        data_test, label_test = [], []
        idx = 0
        for item in pathlist:
            if idx == 100:
                #100类别
                break
            dirlist = os.listdir(item)
            if not (500 <= len(dirlist)):
                continue
            test_idx = 0
            for picpath in dirlist:
                img = cv2.imread(item + '/' + picpath)
                height, width = img.shape[:2]
                target_size = 112
                start_x = (width - target_size) // 2
                start_y = (height - target_size) // 2
                cropped_image = img[start_y:start_y + target_size, start_x:start_x + target_size]
                if test_idx < 100:
                    data_test.append(cropped_image)
                    label_test.append(idx)
                else:
                    data.append(cropped_image)
                    label.append(idx)
                test_idx+=1
            namedict[str(idx)] = item.split('\\')[-1]
            idx += 1
        print("当前有多少人", label[-1] + 1)
        data, label = np.stack(data), np.array(label)
        idx = np.random.permutation(data.shape[0])
        data, label = data[idx], label[idx]

        data_test, label_test = np.stack(data_test), np.array(label_test)
        idx = np.random.permutation(data_test.shape[0])
        data_test, label_test = data_test[idx], label_test[idx]
        trainset = Lacuna100(data, label)
        testset = Lacuna100_test(data_test,label_test)
    return trainset,testset

def dirichlet_split_noniid(train_labels,  client_number=5,alpha=100.0):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''
    # 总类别数、
    n_clients = client_number
    n_classes = train_labels.max()+1#也可以自己手动设置
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # 记录每个类别对应的样本下标
    # 返回二维数组
    class_idcs = [np.argwhere(train_labels==y).flatten()
           for y in range(n_classes)]
    # 定义一个空列表作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    # draw_dataset(classes,train_labels,client_idcs, n_clients)
    return client_idcs

def classify(train_dataset,test_dataset,classifylabel):
    if isinstance(classifylabel,int):
        indices1 = [x for x in range(len(train_dataset)) if train_dataset[x][1] == classifylabel]
        train_imgs = torch.utils.data.Subset(train_dataset, indices1)
        trainlenth = len(indices1)
        indices2 = [x for x in range(len(test_dataset)) if test_dataset[x][1] == classifylabel]
        test_imgs = torch.utils.data.Subset(test_dataset, indices2)
        testlenth = len(indices2)
    
    else:
        indices1 = [x for x in range(len(train_dataset)) if train_dataset[x][1] in classifylabel]
        train_imgs = torch.utils.data.Subset(train_dataset, indices1)
        trainlenth = len(indices1)
        indices2 = [x for x in range(len(test_dataset)) if test_dataset[x][1] in classifylabel]
        test_imgs = torch.utils.data.Subset(test_dataset, indices2)
        testlenth = len(indices2)
    return train_imgs,test_imgs


def classify_one(train_dataset, classifylabel):
    if isinstance(classifylabel, int):
        indices1 = [x for x in range(len(train_dataset)) if train_dataset[x][1] == classifylabel]
        train_imgs = torch.utils.data.Subset(train_dataset, indices1)
    else:
        indices1 = [x for x in range(len(train_dataset)) if train_dataset[x][1] in classifylabel]
        train_imgs = torch.utils.data.Subset(train_dataset, indices1)
    return train_imgs