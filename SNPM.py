import torch
import torch.nn as nn
import DenseASPP
import torchvision
import torch.nn.functional as F

import cv2, math, copy
import matplotlib.pyplot as plt
from collections import OrderedDict


class conv_bn_block(nn.Module):
    def __init__(self, in_channels, num_output, kernel_size, padding=1, dilate_rate=1):
        super(conv_bn_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=num_output, kernel_size=kernel_size, stride=1, padding=padding,
                               dilation=dilate_rate)
        self.bn = nn.BatchNorm2d(num_output)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VIFEM(nn.Module):
    def __init__(self, in_channels, num_output, dilate_rates):
        super(VIFEM, self).__init__()
        self.num_output = num_output
        self.dilate_rates = dilate_rates

        self.conv_a1 = conv_bn_block(in_channels=in_channels, num_output=num_output , kernel_size=3,
                                     padding=dilate_rates[0], dilate_rate=dilate_rates[0])
        self.conv_b1 = conv_bn_block(in_channels=in_channels, num_output=num_output , kernel_size=3,
                                     padding=dilate_rates[0] * 2, dilate_rate=dilate_rates[0] * 2)
        self.conv_c1 = conv_bn_block(in_channels=in_channels, num_output=num_output , kernel_size=1,padding=0)
        self.conv_cat_1 = conv_bn_block(in_channels=num_output*3, num_output=num_output , kernel_size=3)

        self.conv_a2 = conv_bn_block(in_channels=in_channels + num_output  , num_output=num_output ,
                                     kernel_size=3, padding=dilate_rates[1], dilate_rate=dilate_rates[1])
        self.conv_b2 = conv_bn_block(in_channels=in_channels + num_output  , num_output=num_output ,
                                     kernel_size=3, padding=dilate_rates[1] * 2,
                                     dilate_rate=dilate_rates[1] * 2)
        self.conv_c2 = conv_bn_block(in_channels=in_channels+num_output, num_output=num_output , kernel_size=1,padding=0)
        self.conv_cat_2 = conv_bn_block(in_channels=num_output*3, num_output=num_output , kernel_size=3)


        self.conv_res = conv_bn_block(in_channels=num_output, num_output=num_output,
                                      kernel_size=1, padding=0)


    def forward(self, input_tensor):
        x0 = input_tensor
        ax1 = self.conv_a1(input_tensor)
        bx1 = self.conv_b1(input_tensor)
        cx1 = self.conv_c1(input_tensor)
        x1 = torch.cat([ax1, bx1,cx1], dim=1)
        x1 = self.conv_cat_1(x1)

        x1 = torch.cat([x1, x0], dim=1)

        ax2 = self.conv_a2(x1)
        bx2 = self.conv_b2(x1)
        cx2 = self.conv_c2(x1)
        x2 = torch.cat([ax2, bx2,cx2], dim=1)
        x2 = self.conv_cat_2(x2)



        refined_x = self.conv_res(x2)
        return refined_x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)





class SNPM(nn.Module):
    def __init__(self, in_channels=1024, num_output=512, k=2, featuremap_size=24, dilate_rates=[1, 2, 3]):
        super(SNPM, self).__init__()
        self.k = k  # k=2
        self.DenseASPP  = DenseASPP._DenseASPPBlock(1024,512,512)

        self.featuremap_size = featuremap_size  # 24
        self.dilate_rates = dilate_rates
        self.maxpool_1 = nn.MaxPool2d(kernel_size=k, stride=k)
        # self.in_channels = in_channels
        self.conv_att1 = conv_bn_block(in_channels=in_channels, num_output=num_output, kernel_size=3, padding=1)
        self.conv_att2 = conv_bn_block(in_channels=num_output, num_output=num_output, kernel_size=3, padding=1)
        self.conv_att = nn.Conv2d(in_channels=num_output, out_channels=k * k, kernel_size=1, padding=0)
        self.sigmoid_1 = nn.Sigmoid()


        self.aspp = VIFEM(in_channels, num_output, dilate_rates=[1, 2, 3])
        self.conv_fusion = conv_bn_block(in_channels=1536, num_output=2048, kernel_size=3,
                                         padding=1)

    def forward(self, input_tensor):

        if self.k == 1:
            xs = [input_tensor]
        else:
            xs = self.split(input_tensor)

        x = self.maxpool_1(input_tensor)  # x:72*2048*12*12
        x = self.conv_att1(x)  # 72*512*12*12
        x = self.conv_att2(x)  # 72*512*12*12
        x = self.sigmoid_1(self.conv_att(x))  # 72*4*12*12
        if self.k == 1:
            attentions = [x]
        else:
            attentions = self.split_channel(x)

        XS = []
        for i, x in enumerate(xs):

            refined_x = self.aspp(x)

            refined_x = torch.mul(refined_x, attentions[i])
            XS.append(refined_x)
        if self.k == 1:
            output_tensor = XS[0]
        else:
            output_tensor = self.combine(XS)
        output_tensor = torch.cat([input_tensor, output_tensor], dim=1)
        output_tensor = self.conv_fusion(output_tensor)

        return output_tensor

    def split(self, inputs):
        k_height, k_width = self.k, self.k
        height, width = self.featuremap_size, self.featuremap_size
        stride_height = height // k_height
        stride_width = width // k_width

        featuremaps = []
        for i in range(self.k):
            for j in range(self.k):
                featuremaps.append(
                    inputs[:, :, i * stride_height:(i + 1) * stride_height, j * stride_width:(j + 1) * stride_width
                    ])
        return featuremaps

    def split_channel(self, inputs):

        featuremaps = []
        for i in range(self.k * self.k):
            featuremaps.append(inputs[:, i:i + 1, :, :])
        return featuremaps

    def combine(self, inputs):
        origin_featuremaps = []
        for i in range(self.k):
            origin_featuremaps.append(torch.cat(inputs[i * self.k:(i + 1) * self.k], dim=3))
        origin_featuremap = torch.cat(origin_featuremaps, dim=2)

        return origin_featuremap

