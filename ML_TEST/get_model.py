import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from typing import Callable, Any, List
 
import torch
import torch.nn as nn
from torch import Tensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self,num_classes =10):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(28*28, 300)
        self.layer2 = nn.Linear(300, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        out_layer1 = torch.relu(self.layer1(x))
        out_layer2 = torch.relu(self.layer2(out_layer1))
        x = self.classifier(out_layer2)
        out_layer3 = None
        out_layer4 = None
        out_layer5 = None
        return x
    def classify(self, images):
        concatenated = self.forward(images)[0]
        _, max_index = torch.max(concatenated, dim=1)
        return max_index

class LeNet5(nn.Module):
    def __init__(self,num_classes =10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道1，输出通道6，卷积核大小5
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，窗口大小2，步长2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 输出层，10个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # 展平
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BasicBlock(nn.Module):
    # 定义基本的残差块
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.conv_layer1 = self._make_conv_1(3,64)
        self.conv_layer2 = self._make_conv_1(64,128)
        self.conv_layer3 = self._make_conv_2(128,256)
        self.conv_layer4 = self._make_conv_2(256,512)
        self.conv_layer5 = self._make_conv_2(512,512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),    # 这里修改一下输入输出维度
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
            # 使用交叉熵损失函数，pytorch的nn.CrossEntropyLoss()中已经有过一次softmax处理，这里不用再写softmax
        )

    def _make_conv_1(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        return layer
    def _make_conv_2(self,in_channels,out_channels):
        layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
              )
        return layer

    def forward(self, x):
        # 32*32 channel == 3
        x = self.conv_layer1(x)
        # 16*16 channel == 64
        x = self.conv_layer2(x)
        # 8*8 channel == 128
        x = self.conv_layer3(x)
        # 4*4 channel == 256
        x = self.conv_layer4(x)
        # 2*2 channel == 512
        x = self.conv_layer5(x)
        # 1*1 channel == 512
        x = x.view(x.size(0), -1)
        # 512
        x = self.classifier(x)
        # 10
        return x


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
 
 
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):# 第一个参数是输入的通道数，第二个是增长率是一个重要的超参数，它控制了每个密集块中特征图的维度增加量，
        #                第四个参数是Dropout正则化上边的概率
        super(_DenseLayer, self).__init__()# 调用父类的构造方法，这句话的意思是在调用nn.Sequential的构造方法
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),  # 批量归一化
        self.add_module('relu1', nn.ReLU(inplace=True)),     # ReLU层
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),    # 表示其输出为4*k   其中bn_size等于4，growth_rate为k     不改变大小，只改变通道的个数
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),  # 批量归一化
        self.add_module('relu2', nn.ReLU(inplace=True)),         # 激活函数
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),    # 输出为growth_rate：表示输出通道数为k  提取特征
        self.drop_rate = drop_rate
 
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # 通道维度连接
 
 
class _DenseBlock(nn.Sequential):  # 构建稠密块
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate): # 密集块中密集层的数量，第二参数是输入通道数量
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
 
 
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):# 输入通道数 输出通道数
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
 
 
# DenseNet网络模型基本结构
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=4):
 
        super(DenseNet, self).__init__()
 
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
 
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
 
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
 
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
 
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
 
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
 
 
def densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model

 

def return_model(params):
    dataset = params.model_name
    if params.model_name == "Lenet5":
        return LeNet5(params.num_class)
    if params.model_name == "VGG11":
        return VGG11(num_classes=params.num_class)
    if params.model_name == "ResNet":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=params.num_class)
    if params.model_name == "MobileNetV2":
        # return MobileNetV2(num_classes= params.num_class)
        return torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    if params.model_name == "DenseNet":
        return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),num_classes=params.num_class)