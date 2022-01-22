import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyModel,GaussianConditional,EntropyBottleneck
from compressai.models import CompressionModel
from compressai.layers import AttentionBlock
import sys, os, math, time
sys.path.append('..')
import threading
import queue
import torchac
        
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
# each module should include encoding/decoding time
class RecProbModel(CompressionModel):

    def __init__(
        self,
        channels,
    ):
        super().__init__(channels)

        self.channels = int(channels)
        
        self.sigma = self.mu = self.prior_latent = None
        self.RPM = RPM(channels)
        self.gaussian_conditional = GaussianConditional(None)
        
    def set_RPM(self, RPM_flag):
        self.RPM_flag = RPM_flag
        
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def loss(self):
        if self.RPM_flag:
            return torch.FloatTensor([0]).squeeze(0).cuda(0)
        return self.aux_loss()

    def forward(
        self, x, rpm_hidden, training = None, prior_latent=None
    ):
        if self.RPM_flag:
            assert prior_latent is not None, 'prior latent is none!'
            self.sigma, self.mu, rpm_hidden = self.RPM(prior_latent, rpm_hidden.to(x.device))
            self.sigma = torch.maximum(self.sigma, torch.FloatTensor([-7.0]).to(x.device))
            self.sigma = torch.exp(self.sigma)/10
            x_hat,likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
            rpm_hidden = rpm_hidden
        else:
            x_hat,likelihood = self.entropy_bottleneck(x,training=training)
        prior_latent = torch.round(x).detach()
        return x_hat, likelihood, rpm_hidden.detach(), prior_latent
        
    def get_actual_bits(self, string):
        bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        # log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
        # bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
        bits_est = torch.sum(torch.clamp(-1.0 * torch.log(likelihoods + 1e-5) / math.log(2.0), 0, 50))
        return bits_est
        
    def compress(self, x):
        if self.RPM_flag:
            indexes = self.gaussian_conditional.build_indexes(self.sigma)
            string = self.gaussian_conditional.compress(x, indexes, means=self.mu)
        else:
            string = self.entropy_bottleneck.compress(x)
        return string

    def decompress(self, string, shape):
        if self.RPM_flag:
            indexes = self.gaussian_conditional.build_indexes(self.sigma)
            x_hat = self.gaussian_conditional.decompress(string, indexes, means=self.mu)
        else:
            x_hat = self.entropy_bottleneck.decompress(string, shape)
        return x_hat
        
    # we should only use one hidden from compression or decompression
    def compress_slow(self, x, rpm_hidden, prior_latent):
        # shouldnt be used together with forward()
        # otherwise rpm_hidden will be messed up
        self.eAC_t = self.eNet_t = 0
        shape = x.size()[-2:]
        if self.RPM_flag:
            assert prior_latent is not None, 'prior latent is none!'
            # network part
            t_0 = time.perf_counter()
            sigma, mu, rpm_hidden = self.RPM(prior_latent, rpm_hidden.to(prior_latent.device))
            sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(sigma.device))
            sigma = torch.exp(sigma)/10
            self.eNet_t += time.perf_counter() - t_0
            # ac part
            t_0 = time.perf_counter()
            indexes = self.gaussian_conditional.build_indexes(sigma)
            string = self.gaussian_conditional.compress(x, indexes, means=mu)
            x_hat,_ = self.gaussian_conditional(x, sigma, means=mu, training=self.training)
            self.eAC_t += time.perf_counter() - t_0
        else:
            t_0 = time.perf_counter()
            string = self.entropy_bottleneck.compress(x)
            x_hat,_ = self.entropy_bottleneck(x,training=self.training)
            self.eNet_t += 0
            self.eAC_t += time.perf_counter() - t_0
        prior_latent = torch.round(x_hat).detach()
        self.enc_t = self.eNet_t + self.eAC_t
        return x_hat, string, rpm_hidden.detach(), prior_latent
        
    def decompress_slow(self, string, shape, rpm_hidden, prior_latent):
        self.dAC_t = self.dnet_t = 0
        if self.RPM_flag:
            assert prior_latent is not None, 'prior latent is none!'
            # NET
            t_0 = time.perf_counter()
            sigma, mu, rpm_hidden = self.RPM(prior_latent, rpm_hidden.to(prior_latent.device))
            sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(sigma.device))
            sigma = torch.exp(sigma)/10
            self.dnet_t += time.perf_counter() - t_0
            # AC
            t_0 = time.perf_counter()
            indexes = self.gaussian_conditional.build_indexes(sigma)
            x_hat = self.gaussian_conditional.decompress(string, indexes, means=mu)
            self.dAC_t += time.perf_counter() - t_0
        else:
            t_0 = time.perf_counter()
            x_hat = self.entropy_bottleneck.decompress(string, shape)
            self.dnet_t += 0
            self.dAC_t += time.perf_counter() - t_0
        prior_latent = torch.round(x_hat).detach()
        self.dec_t = self.dnet_t + self.dAC_t
        return x_hat, rpm_hidden.detach(), prior_latent
        
class MeanScaleHyperPriors(CompressionModel):

    def __init__(
        self,
        channels,
        entropy_trick=True,
    ):
        super().__init__(channels)

        self.channels = int(channels)
        
        self.sigma = self.mu = self.z_string = None
        self.gaussian_conditional = GaussianConditional(None)
        
        self.h_a1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        
        self.h_a2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        
        self.h_s1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        
        self.h_s2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
        )
        
        self.scale_table = get_scale_table()

        self.entropy_trick = entropy_trick
        
    def update(self, scale_table=None, force=False):
        updated = self.gaussian_conditional.update_scale_table(self.scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def loss(self):
        return self.aux_loss()

    def forward(
        self, x, training = None
    ):
        z = self.h_a1(x)
        z = self.h_a2(z)
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        
        self.z = z # for fast compression
            
        g = self.h_s1(z_hat)
        gaussian_params = self.h_s2(g)
            
        self.sigma, self.mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        # post-process sigma to stablize training
        self.sigma = torch.maximum(self.sigma, torch.FloatTensor([-7.0]).to(x.device))
        self.sigma = torch.exp(self.sigma)
        x_hat,x_likelihood = self.gaussian_conditional(x, self.sigma, means=self.mu, training=training)
        return x_hat, (x_likelihood,z_likelihood)
        
    def get_actual_bits(self, string):
        (x_string,z_string) = string
        x_act = torch.FloatTensor([len(s)*8 for s in x_string])
        z_act = torch.FloatTensor([len(s)*8 for s in z_string])
        bits_act = x_act + z_act
        return bits_act
        
    def get_estimate_bits(self, likelihoods):
        (x_likelihood,z_likelihood) = likelihoods
        log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(x_likelihood.device)
        bs = x_likelihood.size(0)
        x_est = torch.sum(torch.log(x_likelihood.view(bs,-1)),dim=-1) / (-log2)
        z_est = torch.sum(torch.log(z_likelihood.view(bs,-1)),dim=-1) / (-log2)
        bits_est = x_est + z_est
        return bits_est
        
    def compress(self, x):
        # a fast implementation of compression
        z_string = self.entropy_bottleneck.compress(self.z)
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_string = self.gaussian_conditional.compress(x, indexes, means=self.mu)
        return (x_string,z_string)

    def decompress(self, string, shape):
        indexes = self.gaussian_conditional.build_indexes(self.sigma)
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=self.mu)
        return x_hat
        
    # we should only use one hidden from compression or decompression
    def compress_slow(self, x, decode=False):
        # shouldnt be used together with forward()
        self.eAC_t = self.eNet_t = 0
        # NET
        t_0 = time.perf_counter()
        B,C,H,W = x.size()
        z = self.h_a1(x)
        z = self.h_a2(z)
        self.eNet_t += time.perf_counter() - t_0
        # AC
        t_0 = time.perf_counter()
        z_hat, _ = self.entropy_bottleneck(z,training=self.training)
        self.eAC_t += time.perf_counter() - t_0
        # NET
        t_0 = time.perf_counter()
        g = self.h_s1(z_hat)
        gaussian_params = self.h_s2(g)
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(x.device))
        sigma = torch.exp(sigma)
        self.eNet_t += time.perf_counter() - t_0
        # AC
        t_0 = time.perf_counter()
        if decode:
            x_hat,_ = self.gaussian_conditional(x, sigma, means=mu, training=self.training)
        else:
            x_hat = None
        # AC
        if self.entropy_trick:
            z = z.permute(1,0,2,3).unsqueeze(0).contiguous()
            z_size = z.size()[-3:]
        else:
            z_size = z.size()[-2:]
        z_string = self.entropy_bottleneck.compress(z)
        
        indexes = self.gaussian_conditional.build_indexes(sigma)
        if self.entropy_trick:
            x = x.permute(1,0,2,3).unsqueeze(0).contiguous()
            indexes = indexes.permute(1,0,2,3).unsqueeze(0).contiguous()
            mu = mu.permute(1,0,2,3).unsqueeze(0).contiguous()
        x_string = self.gaussian_conditional.compress(x, indexes, means=mu)
       
        self.eAC_t += time.perf_counter() - t_0
        self.enc_t = self.eNet_t + self.eAC_t
        return x_hat, (x_string, z_string), z_size
        
    def decompress_slow(self, string, shape):
        # shape?
        self.dAC_t = self.dnet_t = 0
        # AC
        t_0 = time.perf_counter()
        z_hat = self.entropy_bottleneck.decompress(string[1], shape)
        if self.entropy_trick:
            z_hat = z_hat.squeeze(0).permute(1,0,2,3).contiguous()
        self.dAC_t += time.perf_counter() - t_0
        # NET
        t_0 = time.perf_counter()
        g = self.h_s1(z_hat)
        gaussian_params = self.h_s2(g)
        sigma, mu = torch.split(gaussian_params, self.channels, dim=1) # for fast compression
        sigma = torch.maximum(sigma, torch.FloatTensor([-7.0]).to(sigma.device))
        sigma = torch.exp(sigma)
        self.dnet_t += time.perf_counter() - t_0
        # AC
        t_0 = time.perf_counter()
        indexes = self.gaussian_conditional.build_indexes(sigma)
        if self.entropy_trick:
            indexes = indexes.permute(1,0,2,3).unsqueeze(0).contiguous()
            mu = mu.permute(1,0,2,3).unsqueeze(0).contiguous()
        x_hat = self.gaussian_conditional.decompress(string[0], indexes, means=mu)
        if self.entropy_trick:
            x_hat = x_hat.squeeze(0).permute(1,0,2,3).contiguous()
        self.dAC_t += time.perf_counter() - t_0
        self.dec_t = self.dnet_t + self.dAC_t
        return x_hat
        
# conditional probability
# predict y_t based on parameters computed from y_t-1
class RPM(nn.Module):
    def __init__(self, channels=128, act=torch.tanh):
        super(RPM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=1, padding=1)
        self.channels = channels
        self.lstm = ConvLSTM(channels)

    def forward(self, x, hidden):
        # [B,C,H//16,W//16]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        x, hidden = self.lstm(x, hidden.to(x.device))
            
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        sigma_mu = F.relu(self.conv8(x))
        sigma, mu = torch.split(sigma_mu, self.channels, dim=1)
        return sigma, mu, hidden
        
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
        
def test(name = 'Joint'):
    channels = 128
    if name =='RPM':
        net = RecProbModel(channels)
    else:
        net = MeanScaleHyperPriors(channels,useAttention=True)
    x = torch.rand(4, channels, 14, 14)
    import torch.optim as optim
    from tqdm import tqdm
    parameters = set(p for n, p in net.named_parameters())
    optimizer = optim.Adam(parameters, lr=1e-4)
    rpm_hidden = torch.zeros(1,channels*2,14,14)
    isTrain = True
    rpm_flag = True
    if name == 'RPM':
        net.set_prior(x)
            
    train_iter = tqdm(range(0,10000))
    duration_e = duration_d = bits_est = 0
    for i,_ in enumerate(train_iter):
        optimizer.zero_grad()
        
        net.update(force=True)

        if name == 'RPM':
            net.set_RPM(rpm_flag)
            if isTrain:
                x_hat, likelihoods, rpm_hidden = net(x,rpm_hidden,training=True)
                string = net.compress(x)
            else:
                x_q, _, _ = net(x,rpm_hidden,training=False)
                string, _, duration_e = net.compress_slow(x,rpm_hidden)
                x_hat, rpm_hidden, duration_d = net.decompress_slow(string, x.size()[-2:], rpm_hidden)
                net.set_prior(x)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
        elif name == 'Joint':
            if isTrain:
                x_hat, likelihoods = net(x,x,training=True)
                string = net.compress(x)
            else:
                x_q, _ = net(x,x,training=False)
                string, shape, duration_e = net.compress_slow(x, x)
                x_hat, duration_d = net.decompress_slow(string, shape, x)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
        else:
            if isTrain:
                x_hat, likelihoods = net(x,training=True)
                string = net.compress(x)
            else:
                x_q,_ = net(x,training=False)
                string, shape, duration_e = net.compress_slow(x)
                x_hat, duration_d = net.decompress_slow(string, shape)
                mse2 = torch.mean(torch.pow(x_hat-x_q,2))
            
        bits_act = net.get_actual_bits(string)
        mse = torch.mean(torch.pow(x-x_hat,2))*1024
        
        if isTrain:
            bits_est = net.get_estimate_bits(likelihoods)
            loss = bits_est + net.loss() + mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1)
            optimizer.step()
        
            train_iter.set_description(
                f"Batch: {i:4}. "
                f"loss: {float(loss):.2f}. "
                f"bits_est: {float(bits_est):.2f}. "
                f"bits_act: {float(bits_act):.2f}. "
                f"MSE: {float(mse):.2f}. "
                f"ENC: {float(duration_e):.3f}. "
                f"DEC: {float(duration_d):.3f}. ")
        else:
            train_iter.set_description(
                f"Batch: {i:4}. "
                f"bits_act: {float(bits_act):.2f}. "
                f"MSE: {float(mse):.2f}. "
                f"MSE2: {float(mse2):.4f}. "
                f"ENC: {float(duration_e):.3f}. "
                f"DEC: {float(duration_d):.3f}. ")

if __name__ == '__main__':
    seq_len = 6
    channels = 128
    num_workers = 2
    h = w = 16
    

