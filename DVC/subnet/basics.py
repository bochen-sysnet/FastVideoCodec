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


# classes

class LayerNorm3D(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

def FeedForward3D(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        LayerNorm3D(dim),
        nn.Conv3d(dim, dim * mult, 1),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv3d(dim * mult, dim, 1)
    )

# MHRAs (multi-head relation aggregators)

class LocalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        local_aggr_kernel = 5
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        # they use batchnorm for the local MHRA instead of layer norm
        self.norm = nn.BatchNorm3d(dim)

        # only values, as the attention matrix is taking care of by a convolution
        self.to_v = nn.Conv3d(dim, inner_dim, 1, bias = False)

        # this should be equivalent to aggregating by an attention matrix parameterized as a function of the relative positions across each axis
        self.rel_pos = nn.Conv3d(heads, heads, local_aggr_kernel, padding = local_aggr_kernel // 2, groups = heads)

        # combine out across all the heads
        self.to_out = nn.Conv3d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        b, c, *_, h = *x.shape, self.heads

        # to values
        v = self.to_v(x)

        # split out heads
        v = rearrange(v, 'b (c h) ... -> (b c) h ...', h = h)

        # aggregate by relative positions
        out = self.rel_pos(v)

        # combine heads
        out = rearrange(out, '(b c) h ... -> b (c h) ...', b = b)
        return self.to_out(out)

class GlobalMHRA(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = LayerNorm3D(dim)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)

        shape, h = x.shape, self.heads

        x = rearrange(x, 'b c ... -> b c (...)')

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = h), (q, k, v))

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n', h = h)

        out = self.to_out(out)
        return out.view(*shape)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mhsa_type = 'g',
        local_aggr_kernel = 5,
        dim_head = 64,
        ff_mult = 4,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mhsa_type == 'l':
                attn = LocalMHRA(dim, heads = heads, dim_head = dim_head, local_aggr_kernel = local_aggr_kernel)
            elif mhsa_type == 'g':
                attn = GlobalMHRA(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv3d(dim, dim, 3, padding = 1),
                attn,
                FeedForward3D(dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(self, x):
        for dpe, attn, ff in self.layers:
            x = dpe(x) + x
            x = attn(x) + x
            x = ff(x) + x

        return x

# main class

class Uniformer(nn.Module):
    def __init__(
        self,
        *,
        dims = (64, 128, 256, 512),
        depths = (3, 4, 8, 3),
        mhsa_types = ('l', 'l', 'g', 'g'),
        local_aggr_kernel = 5,
        channels = 3,
        ff_mult = 4,
        dim_head = 64,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv3d(channels, init_dim, (1, 1, 1), stride = (1, 1, 1), padding = (0, 0, 0))

        dim_in_out = tuple(zip(dims[:-1], dims[1:]))
        mhsa_types = tuple(map(lambda t: t.lower(), mhsa_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mhsa_type) in enumerate(zip(depths, mhsa_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Transformer(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mhsa_type = mhsa_type,
                    ff_mult = ff_mult,
                    ff_dropout = ff_dropout,
                    attn_dropout = attn_dropout
                ),
                nn.Sequential(
                    LayerNorm3D(stage_dim),
                    nn.Conv3d(stage_dim, dims[ind + 1], (1, 1, 1), stride = (1, 1, 1)),
                ) if not is_last else None
            ]))

    def forward(self, video):
        x = self.to_tokens(video)

        for transformer, conv in self.stages:
            x = transformer(x)

            if exists(conv):
                x = conv(x)

        return x