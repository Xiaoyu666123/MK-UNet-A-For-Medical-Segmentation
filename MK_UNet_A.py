import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def init_weights(module, name=None, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= max(1, module.groups)
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    if act == 'relu6':
        return nn.ReLU6(inplace)
    if act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    if act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    if act == 'gelu':
        return nn.GELU()
    if act == 'hswish':
        return nn.Hardswish(inplace)
    raise NotImplementedError(f'activation layer [{act}] is not found')


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes or in_planes
        ratio = min(ratio, in_planes)
        self.reduced_channels = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.fc2(self.activation(self.fc1(a)))
        m = self.max_pool(x)
        m = self.fc2(self.activation(self.fc1(m)))
        return self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7, 11)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        return self.sigmoid(self.conv(x))


class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super().__init__()

        def _gcd_list(*vals):
            g = 0
            for v in vals:
                g = v if g == 0 else math.gcd(g, int(v))
            return max(1, g)

        if kernel_size == 1:
            actual_groups = 1
        else:
            actual_groups = _gcd_list(F_g, F_l, F_int, groups)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1, padding=kernel_size // 2, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1, padding=kernel_size // 2, groups=actual_groups, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7, activation='relu'):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio, activation=activation)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class MultiKernelDepthWiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride=1, activation='relu6', dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, stride, k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation, inplace=True)
            ) for k in kernel_sizes
        ])
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        outs = []
        for conv in self.dwconvs:
            out = conv(x)
            outs.append(out)
            if not self.dw_parallel:
                x = x + out
        return outs


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True,
                 kernel_sizes=(1, 3, 5), activation='relu6', use_cbam=False):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.in_channel = in_c
        self.out_channel = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = (self.stride == 1)
        self.ex_c = int(self.in_channel * expansion_factor)
        self.use_cbam = use_cbam

        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.mult_scale_dwconv = MultiKernelDepthWiseConv(self.ex_c, self.kernel_sizes, stride=self.stride,
                                                          activation=activation, dw_parallel=dw_parallel)

        if self.add:
            self.combined_channels = self.ex_c
        else:
            self.combined_channels = self.ex_c * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channel)
        )

        if self.use_cbam:
            self.cbam = CBAM(self.out_channel, ratio=16, kernel_size=7)

        if self.use_skip_connection and (self.in_channel != self.out_channel):
            self.conv1x1 = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0, bias=False)

        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        x_in = x
        out = self.pconv1(x)
        dwouts = self.mult_scale_dwconv(out)
        if self.add:
            dout = sum(dwouts)
        else:
            dout = torch.cat(dwouts, dim=1)
        g = gcd(self.combined_channels, self.out_channel) or 1
        dout = channel_shuffle(dout, g)
        out = self.pconv2(dout)

        if self.use_cbam:
            out = self.cbam(out)

        if self.use_skip_connection:
            if self.in_channel != self.out_channel:
                x_in = self.conv1x1(x_in)
            return x_in + out
        return out


def mk_irb_bottleneck(in_channel, out_channel, n=1, s=1, expansion_factor=2,
                      dw_parallel=True, add=True, kernel_sizes=(1, 3, 5), activation='relu6', use_cbam=False):
    blocks = []
    blocks.append(MultiKernelInvertedResidualBlock(in_channel, out_channel, s, expansion_factor,
                                                   dw_parallel, add, kernel_sizes, activation, use_cbam=use_cbam))
    for _ in range(1, n):
        blocks.append(MultiKernelInvertedResidualBlock(out_channel, out_channel, 1, expansion_factor,
                                                       dw_parallel, add, kernel_sizes, activation, use_cbam=use_cbam))
    return nn.Sequential(*blocks)


class BoundaryHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden = hidden_channels or in_channels // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        return self.conv(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x, target_size):
        # 双线性插值调整尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        # 可学习卷积增强边界
        return self.conv(x)

class AB_MKUnet(nn.Module):

    def __init__(self, num_classes=1, in_channels=3, channels=(16, 32, 64, 96, 160), depths=(1, 1, 1, 1, 1),
                 kernel_sizes=(1, 3, 5), expansion_factor=2, gag_kernel=3, cbam_strategy='deep',
                 use_boundary_head=True):
        super().__init__()

        # 根据策略决定哪些层使用 CBAM
        if cbam_strategy == 'none':
            cbam_flags = [False, False, False, False, False]
        elif cbam_strategy == 'shallow':
            cbam_flags = [False, True, True, False, False]
        elif cbam_strategy == 'deep':
            cbam_flags = [False, False, True, True, True]  # 推荐
        elif cbam_strategy == 'all':
            cbam_flags = [False, True, True, True, True]
        else:
            raise ValueError(f"Unknown cbam_strategy: {cbam_strategy}")

        # Encoders
        self.enc1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes, use_cbam=cbam_flags[0])
        self.enc2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes, use_cbam=cbam_flags[1])
        self.enc3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes, use_cbam=cbam_flags[2])
        self.enc4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes, use_cbam=cbam_flags[3])
        self.enc5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes, use_cbam=cbam_flags[4])

        # GAGs
        self.GAG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                         kernel_size=gag_kernel, groups=max(1, channels[3] // 2))
        self.GAG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                         kernel_size=gag_kernel, groups=max(1, channels[2] // 2))
        self.GAG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                         kernel_size=gag_kernel, groups=max(1, channels[1] // 2))
        self.GAG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                         kernel_size=gag_kernel, groups=max(1, channels[0] // 2))

        # Decoders
        self.dec1 = mk_irb_bottleneck(channels[4], channels[3], n=1, s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes)
        self.dec2 = mk_irb_bottleneck(channels[3], channels[2], n=1, s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes)
        self.dec3 = mk_irb_bottleneck(channels[2], channels[1], n=1, s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes)
        self.dec4 = mk_irb_bottleneck(channels[1], channels[0], n=1, s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes)
        self.dec5 = mk_irb_bottleneck(channels[0], channels[0], n=1, s=1, expansion_factor=expansion_factor,
                                      kernel_sizes=kernel_sizes)

        # Output Attention
        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)
        self.SA = SpatialAttention()

        # 上采样卷积层
        self.up1 = UpsampleConv(channels[3], channels[3])
        self.up2 = UpsampleConv(channels[2], channels[2])
        self.up3 = UpsampleConv(channels[1], channels[1])
        self.up4 = UpsampleConv(channels[0], channels[0])
        self.up5 = UpsampleConv(channels[0], channels[0])

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        # 边界检测头
        self.use_boundary_head = use_boundary_head
        if use_boundary_head:
            self.boundary_head = BoundaryHead(channels[0])

        named_apply(partial(init_weights, scheme='normal'), self)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        H, W = x.shape[2], x.shape[3]

        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, 2, 2)
        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, 2, 2)
        e3 = self.enc3(p2)
        p3 = F.max_pool2d(e3, 2, 2)
        e4 = self.enc4(p3)
        p4 = F.max_pool2d(e4, 2, 2)
        e5 = self.enc5(p4)
        p5 = F.max_pool2d(e5, 2, 2)

        out = self.CA1(p5) * p5
        out = self.SA(out) * out
        out = F.relu(self.up1(self.dec1(out), e4.shape[2:]))
        gated = self.GAG1(g=out, x=e4)
        out = out + gated

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(self.up2(self.dec2(out), e3.shape[2:]))
        gated = self.GAG2(g=out, x=e3)
        out = out + gated

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(self.up3(self.dec3(out), e2.shape[2:]))
        gated = self.GAG3(g=out, x=e2)
        out = out + gated

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(self.up4(self.dec4(out), e1.shape[2:]))
        gated = self.GAG4(g=out, x=e1)
        out = out + gated

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(self.up5(self.dec5(out), (H, W)))

        # 分割预测
        pred = self.head(out)
        if pred.shape[2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)

        # 边界预测
        if self.use_boundary_head:
            boundary_pred = self.boundary_head(out)
            if boundary_pred.shape[2:] != (H, W):
                boundary_pred = F.interpolate(boundary_pred, size=(H, W), mode='bilinear', align_corners=True)
            return pred, boundary_pred

        return pred