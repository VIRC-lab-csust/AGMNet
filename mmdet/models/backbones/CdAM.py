import torch
from mmcv.cnn import ConvModule

from __future__ import division
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    
class CoordAtt(nn.Module):
    def __init__(self, inp):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        conv_cfg = dict(type='Conv2d')
        norm_cfg = dict(type='SyncBN', requires_grad=True)

        self.g = ConvModule(
            inp,
            inp,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.conv_out = ConvModule(
            inp,
            inp,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.theta = ConvModule(
            inp,
            inp,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.phi = ConvModule(
            inp,
            inp,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)

        self.ca = ChannelAttention(inp * 4)
        self.sa = SpatialAttention()

        self.conv_mask = nn.Conv2d(inp, 1, kernel_size=1)

    def embedded_gaussian(self, theta_x, phi_x):
        """Embedded gaussian with temperature."""

        # NonLocal2d pairwise_weight: [n, h, w]
        pairwise_weight = torch.matmul(theta_x, phi_x)
            # theta_x.shape[-1] is `self.inter_channels`
        pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.sigmoid()
        return pairwise_weight

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()

        theta_x = self.theta(x)
        phi_x = self.phi(x)
        theta_x -= theta_x.mean(dim=-2, keepdim=True)
        phi_x -= phi_x.mean(dim=-1, keepdim=True)

        x_h = self.pool_h(theta_x).view(n, c, -1).permute(0, 2, 1)#n,h,c
        x_w = self.pool_w(phi_x).view(n, c, -1)#n,c,w

        pairwise_weight = self.embedded_gaussian(x_h, x_w).unsqueeze(1)#n,1,h,w

        unary_mask = self.conv_mask(x)
        unary_mask = unary_mask.sigmoid()#n,1,h,w

        g_x = self.g(x)
        att = (unary_mask + pairwise_weight) * g_x
        att = self.conv_out(att)

        se = self.ca(x) * x
        se = self.sa(se) * se

        y =  att + identity + se
        
        return y