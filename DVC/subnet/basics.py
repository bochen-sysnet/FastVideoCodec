#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
# from .resample2d_package.resample2d import Resample2d
from six.moves import xrange
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from .GDN import GDN
from torch.autograd import Variable
import datetime
from .flowlib import flow_to_image
from einops import rearrange, repeat
from torch import einsum
from math import log, pi

out_channel_N = 64
out_channel_M = 96
# out_channel_N = 128
# out_channel_M = 192
out_channel_mv = 128


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def tensorimwrite(image, name='im'):
    # means = np.array([0.485, 0.456, 0.406])
    # stds = np.array([0.229, 0.224, 0.225])
    import imageio
    if len(image.size()) == 4:
        image = image[0]
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * 255
    imageio.imwrite(name + ".png", image.astype(np.uint8))

def relu(x):
    return x

def yuv_import_444(filename, dims, numfrm, startfrm):
    fp = open(filename, 'rb')
    # fp=open(filename,'rb')

    blk_size = int(dims[0] * dims[1] * 3)
    fp.seek(blk_size * startfrm, 0)
    Y = []
    U = []
    V = []
    # print(dims[0])
    # print(dims[1])
    d00 = dims[0]
    d01 = dims[1]
    # print(d00)
    # print(d01)
    Yt = np.zeros((dims[0], dims[1]), np.int, 'C')
    Ut = np.zeros((d00, d01), np.int, 'C')
    Vt = np.zeros((d00, d01), np.int, 'C')
    print(dims[0])
    YUV = np.zeros((dims[0], dims[1], 3))

    for m in range(dims[0]):
        for n in range(dims[1]):
            Yt[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Ut[m, n] = ord(fp.read(1))
    for m in range(d00):
        for n in range(d01):
            Vt[m, n] = ord(fp.read(1))

    YUV[:, :, 0] = Yt
    YUV[:, :, 1] = Ut
    YUV[:, :, 2] = Vt
    fp.close()
    return YUV


def CalcuPSNR(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))

def MSE2PSNR(MSE):
    return 10 * math.log10(1.0 / (MSE))

def geti(lamb):
    if lamb == 2048:
        return 'H265L20'
    elif lamb == 1024:
        return 'H265L23'
    elif lamb == 512:
        return 'H265L26'
    elif lamb == 256:
        return 'H265L29'
    else:
        print("cannot find lambda : %d"%(lamb))
        exit(0)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # 函数中padding参数可以无视，实际实现的是padding=same的效果
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = torch.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)
                  
# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
        
# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)
        
# attention
def exists(val):
    return val is not None

def attn(q, k, v, mask = None):
    sim = einsum('b i d, b j d -> b i j', q, k)

    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        (q_, k_, v_) = q, k, v
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # attention
        out = attn(q_, k_, v_, mask = mask)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rot_emb(q, k, rot_emb):
    sin, cos = rot_emb
    rot_dim = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :rot_dim], t[..., rot_dim:]), (q, k))
    q, k = map(lambda t: t * cos + rotate_every_two(t) * sin, (q, k))
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)
        self.register_buffer('scales', scales)

    def forward(self, h, w, device):
        scales = rearrange(self.scales, '... -> () ...')
        scales = scales.to(device)

        h_seq = torch.linspace(-1., 1., steps = h, device = device)
        h_seq = h_seq.unsqueeze(-1)

        w_seq = torch.linspace(-1., 1., steps = w, device = device)
        w_seq = w_seq.unsqueeze(-1)

        h_seq = h_seq * scales * pi
        w_seq = w_seq * scales * pi

        x_sinu = repeat(h_seq, 'i d -> i j d', j = w)
        y_sinu = repeat(w_seq, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        return sin, cos

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freqs = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, n, device):
        seq = torch.arange(n, device = device)
        freqs = einsum('i, j -> i j', seq, self.inv_freqs)
        freqs = torch.cat((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, 'n d -> () n d')
        return freqs.sin(), freqs.cos()

class ConvLSTM(nn.Module):
    def __init__(self, channels=128, forget_bias=1.0, activation=F.relu):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Conv2d(2*channels, 4*channels, kernel_size=3, stride=1, padding=1)
        self._forget_bias = forget_bias
        self._activation = activation
        self._channels = channels

    def forward(self, x, state):
        c, h = torch.split(state,self._channels,dim=1)
        x = torch.cat((x, h), dim=1)
        y = self.conv(x)
        j, i, f, o = torch.split(y, self._channels, dim=1)
        f = torch.sigmoid(f + self._forget_bias)
        i = torch.sigmoid(i)
        c = c * f + i * self._activation(j)
        o = torch.sigmoid(o)
        h = o * self._activation(c)

        return h, torch.cat((c, h),dim=1)
        
class BasicBlock(nn.Module):
    def __init__(self, in_planes, kernel_size, padding):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=padding)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, deconv=False):
        super(TransitionBlock, self).__init__()
        self.deconv=deconv
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        if not deconv:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.deconv:
            return out
        else:
            return F.avg_pool2d(out, 2)

class DMBlock(nn.Module):
    def __init__(self, channel):
        super(DMBlock, self).__init__()
        self.l1 = BasicBlock(channel, 1, 0)
        self.l2 = BasicBlock(channel, 3, 1)
        self.l3 = BasicBlock(channel, 1, 0)
        self.l4 = BasicBlock(channel, 3, 1)
        self.aggr = BasicBlock(channel*4, 1, 0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = torch.cat((x1,x2,x3,x4),1)
        out = self.aggr(x5) + x
        return out