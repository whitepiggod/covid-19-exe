# -*- coding: utf-8 -*-
# 比model14不同的是GAP模块，注意力机制中没有加入原始输入
# 这是目前最好的效果，权重在myModel_with_transform_best.pth
# M_Pol----
import torch
import torch.nn as nn
from myTool import ECAResnet
import torchvision.models as models
import torch.nn.functional as F
from attentionBlock import AttentionCha, AttentionSpa


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GAP(nn.Module):
    def  __init__(self, in_ch, out_ch, k_size=3):
        super(GAP, self).__init__()
        self.deConv = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = BasicConv(in_ch, out_ch, 1, 1, 0, 1)
        self.relu = nn.ReLU(True)
        self.atten_spa = AttentionSpa()
        self.atten_cha = AttentionCha(in_ch, out_ch)


    def forward(self, x):
        x2 = self.atten_spa(x)
        x3 = self.atten_cha(x)
        out = self.relu(x2 + x3 + x)
        return out


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCA_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCA_1, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv1x1 = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        res = self.conv1x1(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + res)
        return x

class GCA_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCA_2, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )

        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        res = self.conv_res(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))

        x = self.relu(x_cat + res)
        return x


class DAF(nn.Module):
    def __init__(self, channels_high, channels_low, kernel_size=3, upsample=True):
        super(DAF, self).__init__()
        self.deConv = nn.ConvTranspose2d(channels_high, channels_high, 2, stride=2)
        self.gap = GAP(channels_high // 2, channels_high)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(channels_high, channels_low)
        self.conv2 = conv3x3(channels_high, channels_low)


    def forward(self, fms_high, fms_low):
        x1 = self.deConv(fms_high)
        x1 = self.conv2(x1)
        x2 = self.gap(fms_low)
        x = torch.cat((x1, x2), 1)
        x = self.relu(x)
        out = self.conv1(x)
        return out


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class M_pol(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(M_pol, self).__init__()
        self.pol1 = nn.AvgPool2d(kernel_size=2, stride=2,
                             ceil_mode=True, count_include_pad=False)
        self.pol2 = nn.AvgPool2d(kernel_size=3, stride=2,
                             ceil_mode=True, count_include_pad=False)
        self.pol3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=1,
                                 ceil_mode=True, count_include_pad=False)
        self.pol4 = nn.AvgPool2d(kernel_size=6, stride=2, padding=2,
                                 ceil_mode=True, count_include_pad=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv1x1(channel_in, channel_out)


    def forward(self, x, other):
        x = self.conv(x)
        x1 = self.pol1(x)
        x2 = self.pol2(x)
        x3 = self.pol3(x)
        x4 = self.pol4(x)
        # print(x1.size(), x2.size(), x3.size(), x4.size(), other.size())
        out = self.relu(x1 + x2 + x3 + x4 + other)
        return out


class MyNet(nn.Module):
    def __init__(self, n_class=1):
        super(MyNet, self).__init__()
        # ---- ResNet Backbone ----
        self.ecaresnet = ECAResnet()
        # ---- Receptive Field Block like module ----

        self.gca1 = GCA_1(256, 128)
        self.gca2 = GCA_1(512, 256)
        self.gca3 = GCA_1(1024, 512)
        self.gca4 = GCA_2(2048, 1024)

        bottom_ch = 1024
        self.daf3 = DAF(bottom_ch, 512)
        self.daf2 = DAF(bottom_ch // 2, 256)
        self.daf1 = DAF(bottom_ch // 4, 128)

        self.conv1_1 = conv1x1(128, 1)
        self.conv1_2 = conv1x1(256, 1)
        self.conv1_3 = conv1x1(512, 1)
        self.conv1_4 = conv1x1(1024, 1)

        self.pol1 = M_pol(64, 256)
        self.pol2 = M_pol(256, 512)
        self.pol3 = M_pol(512, 1024)


        if self.training:
            self.initialize_weights()


    def forward(self, x):
        x = self.ecaresnet.conv1(x)        # 64, 176, 176
        pol = x
        x = self.ecaresnet.bn1(x)

        x = self.ecaresnet.relu(x)

        # ---- low-level features ----
        x = self.ecaresnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.ecaresnet.layer1(x)      # bs, 256, 88, 88
        x1 = self.pol1(pol, x1)
        pol = x1

        # ---- high-level features ----
        x2 = self.ecaresnet.layer2(x1)     # bs, 512, 44, 44
        x2 = self.pol2(pol, x2)
        pol = x2
        x3 = self.ecaresnet.layer3(x2)     # bs, 1024, 22, 22
        x3 = self.pol3(pol, x3)
        x4 = self.ecaresnet.layer4(x3)     # bs, 2048, 11, 11

        x1_gca = self.gca1(x1)        # 256 -> 128
        x2_gca = self.gca2(x2)        # 512 -> 256
        x3_gca = self.gca3(x3)        # 1024 -> 512
        x4_gca = self.gca4(x4)        # 2048 -> 1024

        x3 = self.daf3(x4_gca, x3_gca)  # 1/16
        x2 = self.daf2(x3, x2_gca)  # 1/8
        x1 = self.daf1(x2, x1_gca)  # 1/4

        map_1 = self.conv1_1(x1)
        map_2 = self.conv1_2(x2)
        map_3 = self.conv1_3(x3)
        map_4 = self.conv1_4(x4_gca)

        lateral_map_4 = F.interpolate(map_4, scale_factor=32, mode='bilinear')
        lateral_map_3 = F.interpolate(map_3, scale_factor=16, mode='bilinear')
        lateral_map_2 = F.interpolate(map_2, scale_factor=8, mode='bilinear')
        lateral_map_1 = F.interpolate(map_1, scale_factor=4, mode='bilinear')

        return lateral_map_1

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True) #If True, the pre-trained resnet50 will be loaded.
        pretrained_dict = res50.state_dict()
        model_dict = self.ecaresnet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.ecaresnet.load_state_dict(model_dict)



