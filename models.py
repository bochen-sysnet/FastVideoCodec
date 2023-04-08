from __future__ import print_function
import os
import io
import sys
import time
import math
import random
import numpy as np
import subprocess as sp
import shlex
import cv2

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torchvision import transforms
from compressai.layers import GDN, AttentionBlock, ResidualBlock
from entropy_models import RecProbModel,MeanScaleHyperPriors,RPM
from compressai.models.waseda import Cheng2020Attention
import pytorch_msssim
from PIL import Image
import torchac
import compressai

def get_codec_model(name, loss_type='P', compression_level=2, noMeasure=True, use_split=True):
    if name in ['RLVC','DVC','RLVC2']:
        model_codec = IterPredVideoCodecs(name,loss_type=loss_type,compression_level=compression_level,noMeasure=noMeasure,use_split=use_split)
    elif name in ['DVC-pretrained']:
        model_codec = get_DVC_pretrained(compression_level)
    elif 'LSVC' in name:
        model_codec = LSVC(name,loss_type=loss_type,compression_level=compression_level,use_split=use_split)
    elif 'Base' in name:
        model_codec = Base(name, loss_type='P', compression_level=compression_level)
    elif 'SSF-Official' in name:
        model_codec = compressai.zoo.ssf2020(compression_level+1, metric='mse', pretrained=True, progress=True)
        model_codec.name = 'SSF-Official'
        model_codec.compression_level = compression_level
        model_codec.loss_type = loss_type
        init_training_params(model_codec)
    elif 'ELFVC' in name:
        model_codec = ELFVC(name, loss_type='P', compression_level=compression_level)
    else:
        print('Cannot recognize codec:', name)
        exit(1)
    return model_codec
            
def init_training_params(model):
    model.r_img, model.r_bpp, model.r_aux = 1,1,1
    model.stage = 'REC'
    
    psnr_list = [256,512,1024,2048,4096,8192,16384,16384*2]
    msssim_list = [8,16,32,64]
    I_lvl_list = [37,32,27,22,17,12,7,2]
    model.r = psnr_list[model.compression_level] if model.loss_type == 'P' else msssim_list[model.compression_level]
    model.I_level = I_lvl_list[model.compression_level] # [37,32,27,22] poor->good quality
    # print(f'MSE/MSSSIM multiplier:{model.r}, BPG level:{model.I_level}, channels:{model.channels}')
    
    model.fmt_enc_str = "{0:.3f} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}"
    model.fmt_dec_str = "{0:.3f} {1:.3f} {2:.3f}"
    model.meters = {'E-FL':AverageMeter(),'E-MV':AverageMeter(),'eEMV':AverageMeter(),
                    'E-MC':AverageMeter(),'E-RES':AverageMeter(),'eERES':AverageMeter(),
                    'E-NET':AverageMeter(),
                    'D-MV':AverageMeter(),'eDMV':AverageMeter(),'D-MC':AverageMeter(),
                    'D-RES':AverageMeter(),'eDRES':AverageMeter(),'D-NET':AverageMeter()}
    model.bitscounter = {'M':AverageMeter(),'R':AverageMeter()}
        
def compress_whole_video(name, raw_clip, Q, width=256,height=256, GOP=23):
    imgByteArr = io.BytesIO()
    fps = 25
    #Q = 27#15,19,23,27
    output_filename = f'tmp/videostreams/{name}.mp4'
    if name == 'x265-veryfast':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" {output_filename}'
    elif name == 'x265-medium':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset medium -x265-params "crf={Q}:keyint={GOP}:verbose=1" {output_filename}'
    elif name == 'x265-veryslow':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -preset veryslow -x265-params "crf={Q}:keyint={GOP}:verbose=1" {output_filename}'
    elif name == 'x264-veryfast':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryfast -tune zerolatency -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {output_filename}'
    elif name == 'x264-medium':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset medium -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {output_filename}'
    elif name == 'x264-veryslow':
        cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -preset veryslow -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug {output_filename}'
    else:
        print('Codec not supported')
        exit(1)
    # bgr24, rgb24, rgb?
    #process = sp.Popen(shlex.split(f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec {libname} -pix_fmt yuv420p -crf 24 {output_filename}'), stdin=sp.PIPE)
    t_0 = time.perf_counter()
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    for img in raw_clip:
        process.stdin.write(np.array(img).tobytes())
    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()
    # compression time
    compt = time.perf_counter() - t_0
    # check video size
    video_size = os.path.getsize(output_filename)*8
    # Use OpenCV to read video
    clip = []
    t_0 = time.perf_counter()
    cap = cv2.VideoCapture(output_filename)
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret != True:break
        clip.append(transforms.ToTensor()(img))
    # When everything done, release the video capture object
    cap.release()
    # decompression time
    decompt = time.perf_counter() - t_0
    assert len(clip) == len(raw_clip), f'Clip size mismatch {len(clip)} {len(raw_clip)}'
    # create cache
    psnr_list = [];msssim_list = [];bpp_act_list = []
    bpp = video_size*1.0/len(clip)/(height*width)
    for i in range(len(clip)):
        Y1_raw = transforms.ToTensor()(raw_clip[i]).unsqueeze(0)
        Y1_com = clip[i].unsqueeze(0)
        psnr_list += [PSNR(Y1_raw, Y1_com)]
        msssim_list += [MSSSIM(Y1_raw, Y1_com)]
        bpp_act_list += torch.FloatTensor([bpp])
        
    return psnr_list,msssim_list,bpp_act_list,compt/len(clip),decompt/len(clip)

def next_level(cur):
    nxt = cur + torch.normal(torch.FloatTensor([0]),torch.FloatTensor([.5]))
    nxt = int(torch.round(nxt))
    nxt = min(7,nxt)
    nxt = max(0,nxt)
    return nxt

def parallel_compression(args,model, data, compressI=False, level=0, batch_idx=0):
    all_loss_list = []; 
    img_loss_list = []; bpp_list = []; psnr_list = []; bppres_list = []
    aux_loss_list = []; aux2_loss_list = [];aux3_loss_list = []; aux4_loss_list = [];
    no_batch = len(data.size()) == 4
    if isinstance(model,nn.DataParallel):
        name = f"{model.module.name}-{model.module.compression_level}-{model.module.loss_type}-{os.getpid()}"
        I_level = model.module.I_level
        model_name = model.module.name
        model_r = model.module.r
    else:
        name = f"{model.name}-{model.compression_level}-{model.loss_type}-{os.getpid()}"
        I_level = model.I_level
        model_name = model.name
        model_r = model.r
    if no_batch:
        x_hat, bpp, psnr = I_compression(data[0:1], I_level, model_name=name)
        data[0:1] = x_hat
        if compressI:
            bpp_list += [bpp.to(data.device)]
            psnr_list += [psnr.to(data.device)]
    else:
        for i in range(data.size(0)):
            data[i,0:1], bpp_i, psnr_i = I_compression(data[i,0:1], I_level, model_name=name)
    
    # P compression, not including I frame
    if (no_batch and data.size(0) > 1) or (not no_batch): 
        if model_name in ['SSF-Official'] or 'ELFVC' in model_name:
            GOP_size = data.size(0) if no_batch else data.size(1)
            x_prev = data[0:1] if no_batch else data[:,0]
            x_hat_list = []
            if 'ELFVC' in model_name:
                model.reset()
            for i in range(1,GOP_size):
                x_cur = data[i:i+1] if no_batch else data[:,i]
                x_prev, likelihoods = model.forward_inter(x_cur,x_prev)
                mot_like,res_like = likelihoods["motion"],likelihoods["residual"]
                mot_bits = torch.sum(torch.clamp(-1.0 * torch.log(mot_like["y"] + 1e-5) / math.log(2.0), 0, 50)) + \
                        torch.sum(torch.clamp(-1.0 * torch.log(mot_like["z"] + 1e-5) / math.log(2.0), 0, 50))
                res_bits = torch.sum(torch.clamp(-1.0 * torch.log(res_like["y"] + 1e-5) / math.log(2.0), 0, 50)) + \
                        torch.sum(torch.clamp(-1.0 * torch.log(res_like["z"] + 1e-5) / math.log(2.0), 0, 50))
                bpp = (mot_bits + res_bits) / (x_cur.size(0) * x_cur.size(2) * x_cur.size(3))
                bpp_res = (res_bits) / (x_cur.size(0) * x_cur.size(2) * x_cur.size(3))
                mseloss = torch.mean((x_prev - x_cur).pow(2))

                x_prev = x_prev.detach()
                img_loss_list += [model.r*mseloss.to(data.device)]
                psnr_list += [10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(data.device)]
                bpp_list += [bpp.to(data.device)]
                bppres_list += [(bpp_res).to(data.device)]
                x_hat_list.append(x_prev)

                if 'ELFVC' not in model_name: continue
                loss = model.r*mseloss + bpp
                if model.pred_nc or model.side_channel_nc:
                    pred_err_mean = []
                    pred_norm = 0
                    for pred_err in likelihoods["pred_err"]:
                        pred_err_mean += [pred_err.abs().mean()]
                        pred_norm += torch.norm(pred_err,args.norm) if args.norm > 0 else F.smooth_l1_loss(pred_err, torch.zeros_like(pred_err), reduction='sum')
                    aux_loss_list += [pred_err_mean[0]]
                    aux2_loss_list += [pred_err_mean[1]]
                    loss += args.alpha * pred_norm
                    model.stage = 0
                all_loss_list += [loss]
                Q_err_mean = []
                Q_norm = 0
                for Q_err in likelihoods["Q_err"]:
                    Q_err_mean += [Q_err.abs().mean()]
                    Q_norm += torch.norm(Q_err, args.norm) if args.norm > 0 else F.smooth_l1_loss(Q_err, torch.zeros_like(Q_err), reduction='sum')
                aux3_loss_list += [Q_err_mean[0]]
                aux4_loss_list += [Q_err_mean[1]]
            x_hat = torch.cat(x_hat_list,dim=0)
        elif 'Base' == model_name[:4]:
            B,_,H,W = data.size()
            x_prev = data[0:1]
            x_hat_list = []
            for i in range(1,B):
                x_prev, mseloss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, err = \
                    model(data[i:i+1],x_prev)
                if model.training and model.useER and model.soft2hard:
                    model.s2h_stage = 1
                    _, mseloss1, _, _, _, _, _, _ = \
                        model(data[i:i+1],x_prev)
                    model.s2h_stage = 2
                    _, mseloss2, _, _, _, _, _, _ = \
                        model(data[i:i+1],x_prev)
                    model.s2h_stage = 0
                x_prev = x_prev.detach()
                img_loss_list += [model.r*mseloss]
                bpp_list += [bpp]
                if model.training and model.useER and model.soft2hard:
                    bppres_list += [model.r*(mseloss + mseloss1 + mseloss2)/3]
                else:
                    bppres_list += [(bpp_feature + bpp_z)]
                psnr_list += [10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(data.device)]
                if model.useER:
                    if model.training and model.soft2hard:
                        all_loss_list += [(model.r*(mseloss + mseloss1 + mseloss2)/3 + bpp + err[1])]
                    else:
                        all_loss_list += [(model.r*mseloss + bpp + err[1])]
                    aux_loss_list += [err[0]]
                    aux2_loss_list += [err[1]]
                    aux3_loss_list += [err[2]]
                    aux4_loss_list += [err[3]]
                else:
                    all_loss_list += [(model.r*mseloss + bpp)]
                x_hat_list.append(x_prev)
            x_hat = torch.cat(x_hat_list,dim=0)
        elif model_name in ['DVC','RLVC','RLVC2']:
            B,_,H,W = data.size()
            hidden = model.init_hidden(H,W,data.device)
            mv_prior_latent = res_prior_latent = None
            x_prev = data[0:1]
            x_hat_list = []
            for i in range(1,B):
                if model.training:
                    x_prev,hidden,bpp_est,bpp_res_est,img_loss,aux_loss,bpp_act,psnr,mv_prior_latent,res_prior_latent = \
                        model(x_prev, data[i:i+1], hidden, i>1,mv_prior_latent,res_prior_latent)
                else:
                    x_prev,hidden,bpp_est,img_loss,aux_loss,bpp_act,psnr,mv_prior_latent,res_prior_latent = \
                        model(x_prev, data[i:i+1], hidden, i>1,mv_prior_latent,res_prior_latent)
                x_prev = x_prev.detach()
                all_loss_list += [(model.r*img_loss + bpp_est).to(data.device)]
                img_loss_list += [model.r*img_loss.to(data.device)]
                aux_loss_list += [aux_loss.to(data.device)]
                bpp_list += [bpp_est.to(data.device)]
                psnr_list += [psnr.to(data.device)]
                x_hat_list.append(x_prev)
            x_hat = torch.cat(x_hat_list,dim=0)
        elif model_name in ['DVC-pretrained']:
            B,_,H,W = data.size()
            x_prev = data[0:1]
            x_hat_list = []
            for i in range(1,B):
                x_prev, mseloss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = \
                    model(data[i:i+1],x_prev)
                x_prev = x_prev.detach()
                img_loss_list += [model.r*mseloss.to(data.device)]
                aux_loss_list += [10.0*torch.log(1/warploss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(data.device)]
                bpp_list += [bpp.to(data.device)]
                psnr_list += [10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(data.device)]
                aux2_loss_list += [10.0*torch.log(1/interloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(data.device)]
                x_hat_list.append(x_prev)
            x_hat = torch.cat(x_hat_list,dim=0)
            bppres_list = []
        elif 'LSVC' in model_name:
            B,_,H,W = data.size()
            x_hat, x_mc, x_wp, rec_loss, warp_loss, mc_loss, bpp_res, bpp = model(data.detach())
            img_loss_list = [rec_loss*model.r]
            all_loss_list += [(rec_loss*model.r + bpp).to(data.device)]
            N = B-1
            psnr_list += PSNR(data[1:], x_hat, use_list=True)
            aux2_loss_list += PSNR(data[1:], x_mc, use_list=True)
            aux_loss_list += PSNR(data[1:], x_wp, use_list=True)
            x_hat = torch.cat([data[0:1],x_hat], dim=0)
            for pos in range(N):
                bpp_list += [(bpp).to(data.device)]
                if model.training:
                    bppres_list += [(bpp_res).to(data.device)]

    # aggregate loss
    loss = torch.stack(all_loss_list,dim=0).sum(dim=0) if all_loss_list else 0
    be_loss = torch.stack(bpp_list,dim=0).mean(dim=0).cpu().data.item()
    be_res_loss = torch.stack(bppres_list,dim=0).mean(dim=0).cpu().data.item() if bppres_list else 0
    img_loss = torch.stack(img_loss_list,dim=0).mean(dim=0).cpu().data.item() if all_loss_list else 0
    psnr = torch.stack(psnr_list,dim=0).mean(dim=0).cpu().data.item()
    aux_loss = torch.stack(aux_loss_list,dim=0).mean(dim=0).cpu().data.item() if aux_loss_list else 0
    aux2_loss = torch.stack(aux2_loss_list,dim=0).mean(dim=0).cpu().data.item() if aux2_loss_list else 0
    aux3_loss = torch.stack(aux3_loss_list,dim=0).mean(dim=0).cpu().data.item() if aux3_loss_list else 0
    aux4_loss = torch.stack(aux4_loss_list,dim=0).mean(dim=0).cpu().data.item() if aux4_loss_list else 0
    I_psnr = float(psnr_list[0]) if compressI else 0

    return x_hat,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux2_loss,aux3_loss,aux4_loss
        
def I_compression(Y1_raw, I_level, model_name=''):
    # we can compress with bpg,deepcod ...
    batch_size, _, Height, Width = Y1_raw.shape
    prename = "tmp/frames/prebpg" + model_name
    binname = "tmp/frames/bpg" + model_name
    postname = "tmp/frames/postbpg" + model_name
    raw_img = transforms.ToPILImage()(Y1_raw.squeeze(0))
    raw_img.save(prename + '.jpg')
    pre_bits = os.path.getsize(prename + '.jpg')*8
    os.system('bpgenc -f 444 -m 9 ' + prename + '.jpg -o ' + binname + '.bin -q ' + str(I_level))
    os.system('bpgdec ' + binname + '.bin -o ' + postname + '.jpg')
    post_bits = os.path.getsize(binname + '.bin')*8/(Height * Width * batch_size)
    bpp = torch.FloatTensor([post_bits]).squeeze(0)
    bpg_img = Image.open(postname + '.jpg').convert('RGB')
    Y1_com = transforms.ToTensor()(bpg_img).unsqueeze(0)
    psnr = PSNR(Y1_raw, Y1_com)
    # msssim = MSSSIM(Y1_raw, Y1_com)
    return Y1_com, bpp, psnr
    
def load_state_dict_only(model, state_dict, keyword):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if keyword not in name: continue
        if name in own_state:
            own_state[name].copy_(param)
    
def load_state_dict_whatever(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
             continue
        if name in own_state and own_state[name].size() == param.size():
            own_state[name].copy_(param)
            
def load_state_dict_all(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name.endswith("._offset") or name.endswith("._quantized_cdf") or name.endswith("._cdf_length") or name.endswith(".scale_table"):
             continue
        own_state[name].copy_(param)
    
def PSNR(Y1_raw, Y1_com, use_list=False):
    Y1_com = Y1_com.to(Y1_raw.device)
    log10 = torch.log(torch.FloatTensor([10])).squeeze(0).to(Y1_raw.device)
    if not use_list:
        train_mse = torch.mean(torch.pow(Y1_raw - Y1_com, 2))
        quality = 10.0*torch.log(1/train_mse)/log10
    else:
        b = Y1_raw.size()[0]
        quality = []
        for i in range(b):
            train_mse = torch.mean(torch.pow(Y1_raw[i:i+1] - Y1_com[i:i+1].unsqueeze(0), 2))
            psnr = 10.0*torch.log(1/train_mse)/log10
            quality.append(psnr)
    return quality

def MSSSIM(Y1_raw, Y1_com, use_list=False):
    Y1_com = Y1_com.to(Y1_raw.device)
    if not use_list:
        quality = pytorch_msssim.ms_ssim(Y1_raw, Y1_com)
    else:
        bs = Y1_raw.size()[0]
        quality = []
        for i in range(bs):
            quality.append(pytorch_msssim.ms_ssim(Y1_raw[i].unsqueeze(0), Y1_com[i].unsqueeze(0)))
    return quality

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
    
def get_actual_bits(self, string):
    bits_act = torch.FloatTensor([len(b''.join(string))*8]).squeeze(0)
    return bits_act
        
def get_estimate_bits(self, likelihoods):
    # log2 = torch.log(torch.FloatTensor([2])).squeeze(0).to(likelihoods.device)
    # bits_est = torch.sum(torch.log(likelihoods)) / (-log2)
    bits_est = torch.sum(torch.clamp(-1.0 * torch.log(likelihoods + 1e-5) / math.log(2.0), 0, 50))
    return bits_est

class Coder2D(nn.Module):
    def __init__(self, keyword, in_channels=2, channels=128, kernel=3, padding=1, 
                noMeasure=True, downsample=True, entropy_trick=True):
        super(Coder2D, self).__init__()
        if downsample:
            self.enc_conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv2 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv3 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding)
            self.enc_conv4 = nn.Conv2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, bias=False)
            self.gdn1 = GDN(channels)
            self.gdn2 = GDN(channels)
            self.gdn3 = GDN(channels)
            self.dec_conv1 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv2 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv3 = nn.ConvTranspose2d(channels, channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.dec_conv4 = nn.ConvTranspose2d(channels, in_channels, kernel_size=kernel, stride=2, padding=padding, output_padding=1)
            self.igdn1 = GDN(channels, inverse=True)
            self.igdn2 = GDN(channels, inverse=True)
            self.igdn3 = GDN(channels, inverse=True)
        if keyword in ['RLVC','rpm']:
            # for recurrent sequential model
            self.entropy_bottleneck = RecProbModel(channels)
            self.conv_type = 'rec'
            self.entropy_type = 'rpm'
        elif keyword == 'RLVC2':
            self.conv_type = 'rec'
            self.entropy_type = 'rpm2'
            self.RPM = RPM(channels)
            BitEstimator.get_actual_bits = get_actual_bits
            BitEstimator.get_estimate_bits = get_estimate_bits
            self.entropy_bottleneck = BitEstimator(channels)
        else:
            print('Bottleneck not implemented for:',keyword)
            exit(1)
        # print('Conv type:',self.conv_type,'entropy type:',self.entropy_type)
        self.channels = channels
        if self.conv_type == 'rec':
            self.enc_lstm = ConvLSTM(channels)
            self.dec_lstm = ConvLSTM(channels)
            
        self.downsample = downsample
        self.updated = False
        self.noMeasure = noMeasure
        # include two average meter to measure time
        
    def forward(self, x, rae_hidden=None, rpm_hidden=None, RPM_flag=False, prior=None, prior_latent=None):
        # Time measurement: start
        if not self.noMeasure:
            t_0 = time.perf_counter()
            
        self.realCom = not self.training
        # update only once during testing
        if not self.updated and self.realCom and self.entropy_type != 'rpm2':
            self.entropy_bottleneck.update(force=True)
            self.updated = True
            
        if not self.noMeasure:
            self.enc_t = self.dec_t = 0
        
        # latent states
        if self.conv_type == 'rec':
            state_enc, state_dec = torch.split(rae_hidden.to(x.device),self.channels*2,dim=1) 
            
        # compress
        if self.downsample:
            x = self.gdn1(self.enc_conv1(x))
            x = self.gdn2(self.enc_conv2(x))
            
            if self.conv_type == 'rec':
                x, state_enc = self.enc_lstm(x, state_enc)
                
            x = self.gdn3(self.enc_conv3(x))
            x = self.enc_conv4(x) # latent optical flow
                
            latent = x
        else:
            latent = x
        
        # Time measurement: end
        if not self.noMeasure:
            self.enc_t += time.perf_counter() - t_0
        
        # quantization + entropy coding
        if self.entropy_type == 'rpm2':
            if self.training:
                half = float(0.5)
                noise = torch.empty_like(latent).uniform_(-half, half)
                latent_hat = latent + noise
            else:
                latent_hat = torch.round(latent)
            t_0 = time.perf_counter()
            if RPM_flag:
                assert prior_latent is not None, 'prior latent is none!'
                sigma, mu, rpm_hidden = self.RPM(prior_latent, rpm_hidden)
                mu = torch.zeros_like(sigma)
                sigma = sigma.clamp(1e-5, 1e10)
                gaussian = torch.distributions.laplace.Laplace(mu, sigma)
                likelihoods = gaussian.cdf(latent_hat + 0.5) - gaussian.cdf(latent_hat - 0.5)
            else:
                likelihoods = self.entropy_bottleneck(latent_hat + 0.5) - self.entropy_bottleneck(latent_hat - 0.5)
            self.entropy_bottleneck.enc_t = time.perf_counter() - t_0
            self.entropy_bottleneck.dec_t = time.perf_counter() - t_0
            prior_latent = torch.round(latent).detach()
        else:
            self.entropy_bottleneck.set_RPM(RPM_flag)
            if self.noMeasure:
                latent_hat, likelihoods, rpm_hidden, prior_latent = self.entropy_bottleneck(latent, rpm_hidden, training=self.training, prior_latent=prior_latent)
                if self.realCom:
                    latent_string = self.entropy_bottleneck.compress(latent)
            else:
                _, latent_string, _, _ = self.entropy_bottleneck.compress_slow(latent,rpm_hidden,prior_latent=prior_latent)
                latent_hat, rpm_hidden, prior_latent = self.entropy_bottleneck.decompress_slow(latent_string, latent.size()[-2:], rpm_hidden, prior_latent=prior_latent)
            
        # add in the time in entropy bottleneck
        if not self.noMeasure:
            self.enc_t += self.entropy_bottleneck.enc_t
            self.dec_t += self.entropy_bottleneck.dec_t
        
        # calculate bpp (estimated) if it is training else it will be set to 0
        if self.noMeasure:
            bits_est = self.entropy_bottleneck.get_estimate_bits(likelihoods)
        else:
            bits_est = torch.zeros(latent_hat.size(0)).squeeze(0).to(x.device)
        
        # calculate bpp (actual)
        if self.realCom and self.entropy_type != 'rpm2':
            bits_act = self.entropy_bottleneck.get_actual_bits(latent_string)
        else:
            bits_act = bits_est

        # Time measurement: start
        if not self.noMeasure:
            t_0 = time.perf_counter()
            
        # decompress
        if self.downsample:
            x = latent_hat
            x = self.igdn1(self.dec_conv1(x))
            x = self.igdn2(self.dec_conv2(x))
            
            if self.conv_type == 'rec':
                x, state_dec = self.enc_lstm(x, state_dec)
                
            x = self.igdn3(self.dec_conv3(x))
            hat = self.dec_conv4(x)
        else:
            hat = latent_hat
        
        if self.conv_type == 'rec':
            rae_hidden = torch.cat((state_enc, state_dec),dim=1)
            if rae_hidden is not None:
                rae_hidden = rae_hidden.detach()
        
        # Time measurement: end
        if not self.noMeasure:
            self.enc_t += time.perf_counter() - t_0
            self.dec_t += time.perf_counter() - t_0
        
        # auxilary loss
        aux_loss = self.entropy_bottleneck.loss() if self.entropy_type != 'rpm2' else torch.FloatTensor([0]).to(hat.device)
            
        return hat, rae_hidden, rpm_hidden, bits_act, bits_est, aux_loss, prior_latent
    
def generate_graph(graph_type='default'):
    # 7 nodes, 6 edges
    # the order to iterate graph also matters, leave it now
    # BFS or DFS?
    if graph_type == 'default':
        g = {}
        for k in range(30):
            g[k] = [k+1]
        layers = [[i+1] for i in range(30)] # elements in layers
        parents = {i+1:i for i in range(30)}
    elif graph_type == 'onehop':    
        g = {0:[i+1 for i in range(14)]}
        layers = [[i+1 for i in range(14)]]
        parents = {i+1:0 for i in range(14)}
    elif graph_type == '2layers':
        g = {0:[1,2]}
        layers = [[1,2]] # elements in layers
        parents = {1:0,2:0}
    elif graph_type == '3layers':
        g = {0:[1,4],1:[2,3],4:[5,6]}
        layers = [[1,4],[2,3,5,6]] # elements in layers
        parents = {1:0,4:0,2:1,3:1,5:4,6:4}
    elif graph_type == '4layers':
        # 0
        # 1       8
        # 2   5   9     12
        # 3 4 6 7 10 11 13 14
        g = {0:[1,8],1:[2,5],8:[9,12],2:[3,4],5:[6,7],9:[10,11],12:[13,14]}
        layers = [[1,8],[2,5,9,12],[3,4,6,7,10,11,13,14]] # elements in layers
        parents = {1:0,8:0,2:1,5:1,9:8,12:8,3:2,4:2,6:5,7:5,10:9,11:9,13:12,14:12}
    elif graph_type == '5layers':
        # 0
        # 1                   16
        # 2       9           17          24
        # 3   6   10    13    18    21    25    28
        # 4 5 7 8 11 12 14 15 19 20 22 23 26 27 29 30
        g = {0:[1,16],1:[2,9],16:[17,24],2:[3,6],9:[10,13],17:[18,21],24:[25,28],
            3:[4,5],6:[7,8],10:[11,12],13:[14,15],18:[19,20],21:[22,23],25:[26,27],28:[29,30]}
        layers = [[1,16],[2,9,17,24],[3,6,10,13,18,21,25,28],
                [4,5,7,8,11,12,14,15,19,20,22,23,26,27,29,30]]
        parents = {1:0,16:0,2:1,9:1,17:16,24:16,3:2,6:2,10:9,13:9,18:17,21:17,25:24,28:24,
                4:3,5:3,7:6,8:6,11:10,12:10,14:13,15:13,19:18,20:18,22:21,23:21,26:25,27:25,29:28,30:28}
    else:
        print('Undefined graph type:',graph_type)
        exit(1)
    return g,layers,parents

Backward_tensorGrid = {}

def torch_warp(tensorInput, tensorFlow):
    device_id = tensorInput.device.index
    if str(tensorFlow.size()) not in Backward_tensorGrid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())].to(device_id) + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

def log10(x):
    numerator = torch.log(x)
    denominator = torch.log(10)
    return numerator / denominator


def flow_warp(im, flow):
    warp = torch_warp(im, flow)

    return warp

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature
def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer

class Warp_net(nn.Module):
    def __init__(self, in_channels=6,out_channels=3):
        super(Warp_net, self).__init__()
        channelnum = 64

        self.feature_ext = nn.Conv2d(in_channels, channelnum, 3, padding=1)# feature_ext
        self.f_relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)#c0
        self.conv0_p = nn.AvgPool2d(2, 2)# c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)#c1
        self.conv1_p = nn.AvgPool2d(2, 2)# c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)# c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)# c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)# c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)# c5
        self.conv6 = nn.Conv2d(channelnum, out_channels, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))
        c0 = self.conv0(feature_ext)
        c0_p = self.conv0_p(c0)
        c1 = self.conv1(c0_p)
        c1_p = self.conv1_p(c1)
        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling2(c3)#torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling2(c4)# torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
        c5 = self.conv5(c4_u)
        c5 = self.conv6(c5)
        return c5


class MEBasic(nn.Module):
    '''
    Get flow
    '''
    def __init__(self, layername, inshape=8):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(inshape, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x

class ME_Spynet(nn.Module):
    '''
    Get flow
    '''
    def __init__(self, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([ MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4) ])
        
    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))# , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))#, count_include_pad=False))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat([im1list[self.L - 1 - intLevel], # ref image
                                                                        flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample), # targ image
                                                                        flowfiledsUpsample], 1)) # current flow

        return flowfileds
    
def motioncompensation(warpnet, ref, mv):
    warpframe = flow_warp(ref, mv)
    inputfeature = torch.cat((warpframe, ref), 1)
    prediction = warpnet(inputfeature) + warpframe
    return prediction, warpframe

def TFE(warpnet,x_ref,bs,mv_hat,layers,parents,use_split,detach=False):
    MC_frame_list = [None for _ in range(bs)]
    warped_frame_list = [None for _ in range(bs)]
    # for layers in graph
    # get elements of this layers
    # get parents of all elements above
    for layer in layers:
        ref = [] # reference frame
        diff = [] # motion
        for tar in layer: # id of frames in this layer
            if tar>bs:continue
            parent = parents[tar]
            ref += [x_ref if parent==0 else MC_frame_list[parent-1]] # ref needed for this id
            diff += [mv_hat[tar-1:tar].cuda(1) if use_split else mv_hat[tar-1:tar]] # motion needed for this id
        if ref:
            ref = torch.cat(ref,dim=0)
            if detach:
                ref = ref.detach()
            diff = torch.cat(diff,dim=0)
            MC_frame,warped_frame = motioncompensation(warpnet, ref, diff)
            for i,tar in enumerate(layer):
                if tar>bs:continue
                MC_frame_list[tar-1] = MC_frame[i:i+1]
                warped_frame_list[tar-1] = warped_frame[i:i+1]
    MC_frames = torch.cat(MC_frame_list,dim=0)
    warped_frames = torch.cat(warped_frame_list,dim=0)
    return MC_frames,warped_frames
    
def graph_from_batch(bs,isLinear=False,isOnehop=False):
    if isLinear:
        g,layers,parents = generate_graph('default')
    elif isOnehop:
        g,layers,parents = generate_graph('onehop')
    else:
        # I frame is the only first layer
        if bs <=2:
            g,layers,parents = generate_graph('2layers')
        elif bs <=6:
            g,layers,parents = generate_graph('3layers')
        elif bs <=14:
            g,layers,parents = generate_graph('4layers')
        elif bs <=30:
            g,layers,parents = generate_graph('5layers')
        else:
            print('Batch size not supported yet:',bs)
    return g,layers,parents
    
def refidx_from_graph(g,bs):
    ref_index = [-1 for _ in range(bs)]
    for start in g:
        if start>bs:continue
        for k in g[start]:
            if k>bs:continue
            ref_index[k-1] = start
    return ref_index
        
# DVC,RLVC
# Need to measure time and implement decompression for demo
# cache should store start/end-of-GOP information for the action detector to stop; test will be based on it
class IterPredVideoCodecs(nn.Module):
    def __init__(self, name, channels=128, noMeasure=True, loss_type='P',compression_level=2,use_split=True):
        super(IterPredVideoCodecs, self).__init__()
        self.name = name 
        #self.opticFlow = OpticalFlowNet()
        #self.warpnet = MCNet()
        self.opticFlow = ME_Spynet()
        self.warpnet = Warp_net()
        self.mv_codec = Coder2D(self.name, in_channels=2, channels=channels, kernel=3, padding=1, noMeasure=noMeasure)
        self.res_codec = Coder2D(self.name, in_channels=3, channels=channels, kernel=5, padding=2, noMeasure=noMeasure)
        self.channels = channels
        self.loss_type=loss_type
        self.compression_level=compression_level
        self.use_psnr = loss_type=='P'
        init_training_params(self)
        self.epoch = -1
        self.noMeasure = noMeasure
        
        self.use_split = use_split
        if self.use_split:
            self.split()

    def split(self):
        self.opticFlow.cuda(0)
        self.mv_codec.cuda(0)
        self.warpnet.cuda(1)
        self.res_codec.cuda(1)

    def forward(self, Y0_com, Y1_raw, hidden_states, RPM_flag,mv_prior_latent,res_prior_latent):
        # Y0_com: compressed previous frame, [1,c,h,w]
        # Y1_raw: uncompressed current frame
        batch_size, _, Height, Width = Y1_raw.shape
        # hidden states
        rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden = hidden_states
        # estimate optical flow
        t_0 = time.perf_counter()
        # mv_tensors, l0, l1, l2, l3, l4 = self.opticFlow(Y0_com, Y1_raw)
        mv_tensors = self.opticFlow(Y1_raw, Y0_com)
        if not self.noMeasure:
            self.meters['E-FL'].update(time.perf_counter() - t_0)
        # compress optical flow
        mv_hat,rae_mv_hidden,rpm_mv_hidden,mv_act,mv_est,mv_aux,mv_prior_latent = \
            self.mv_codec(mv_tensors, rae_mv_hidden, rpm_mv_hidden, RPM_flag,prior_latent=mv_prior_latent)
        if not self.noMeasure:
            self.meters['E-MV'].update(self.mv_codec.enc_t)
            self.meters['D-MV'].update(self.mv_codec.dec_t)
            self.meters['eEMV'].update(self.mv_codec.entropy_bottleneck.enc_t)
            self.meters['eDMV'].update(self.mv_codec.entropy_bottleneck.dec_t)
        # motion compensation
        t_0 = time.perf_counter()
        # replace
        Y1_MC, Y1_warp = motioncompensation(self.warpnet, Y0_com, mv_hat.cuda(1) if self.use_split else mv_hat)
        t_comp = time.perf_counter() - t_0
        if not self.noMeasure:
            self.meters['E-MC'].update(t_comp)
            self.meters['D-MC'].update(t_comp)
        res_tensor = Y1_raw.to(Y1_MC.device) - Y1_MC
        res_hat,rae_res_hidden,rpm_res_hidden,res_act,res_est,res_aux,res_prior_latent = \
            self.res_codec(res_tensor, rae_res_hidden, rpm_res_hidden, RPM_flag,prior_latent=res_prior_latent)
        if not self.noMeasure:
            self.meters['E-RES'].update(self.res_codec.enc_t)
            self.meters['D-RES'].update(self.res_codec.dec_t)
            self.meters['eERES'].update(self.res_codec.entropy_bottleneck.enc_t)
            self.meters['eDRES'].update(self.res_codec.entropy_bottleneck.dec_t)
        # reconstruction
        Y1_com = torch.clip(res_hat + Y1_MC, min=0, max=1)
        # record time
        self.encoding_time = self.meters['E-FL'].avg + self.meters['E-MV'].avg + self.meters['E-MC'].avg + self.meters['E-RES'].avg + self.meters['D-MV'].avg + self.meters['D-MC'].avg + self.meters['D-RES'].avg
        self.decoding_time = self.meters['D-MV'].avg + self.meters['D-MC'].avg + self.meters['D-RES'].avg
        ##### compute bits
        # estimated bits
        bpp_est = (mv_est + res_est.to(mv_est.device))/(Height * Width * batch_size)
        bpp_res_est = (res_est)/(Height * Width * batch_size)
        # actual bits
        bpp_act = (mv_act + res_act.to(mv_act.device))/(Height * Width * batch_size)
        # auxilary loss
        aux_loss = (mv_aux if self.stage != 'RES' else mv_aux.detach()) + \
                    (res_aux.to(mv_aux.device).detach() if self.stage == 'MC' else res_aux.to(mv_aux.device))/2
        # calculate metrics/loss
        psnr = PSNR(Y1_raw, Y1_com.to(Y1_raw.device))
        img_loss = torch.mean((Y1_raw - Y1_com.to(Y1_raw.device)).pow(2))
        # hidden states
        hidden_states = (rae_mv_hidden.detach(), rae_res_hidden.detach(), rpm_mv_hidden, rpm_res_hidden)
        if self.training:
            return Y1_com.to(Y1_raw.device), hidden_states, bpp_est, bpp_res_est, img_loss, aux_loss, bpp_act, psnr, mv_prior_latent, res_prior_latent
        else:
            return Y1_com.to(Y1_raw.device), hidden_states, bpp_est, img_loss, aux_loss, bpp_act, psnr, mv_prior_latent, res_prior_latent
    
    def init_hidden(self, h, w, device):
        rae_mv_hidden = torch.zeros(1,self.channels*4,h//4,w//4)
        rae_res_hidden = torch.zeros(1,self.channels*4,h//4,w//4)
        rpm_mv_hidden = torch.zeros(1,self.channels*2,h//16,w//16)
        rpm_res_hidden = torch.zeros(1,self.channels*2,h//16,w//16)
        rae_mv_hidden = rae_mv_hidden.to(device)
        rae_res_hidden = rae_res_hidden.to(device)
        rpm_mv_hidden = rpm_mv_hidden.to(device)
        rpm_res_hidden = rpm_res_hidden.to(device)
        return (rae_mv_hidden, rae_res_hidden, rpm_mv_hidden, rpm_res_hidden)
        
def TreeFrameReconForward(warpnet,res_codec,x,bs,mv_hat,layers,parents,mode='forward'):
    x_tar = x[1:]
    MC_frame_list = [None for _ in range(bs)]
    warped_frame_list = [None for _ in range(bs)]
    com_frame_list = [None for _ in range(bs)]
    res_act_list = [None for _ in range(bs)]
    res_est_list = [None for _ in range(bs)]
    # for layers in graph
    # get elements of this layers
    # get parents of all elements above
    for layer in layers:
        ref = [] # reference frame
        diff = [] # motion
        target = [] # target frames
        for tar in layer: # id of frames in this layer
            if tar>bs:continue
            parent = parents[tar]
            ref += [x[:1] if parent==0 else com_frame_list[parent-1]] # ref needed for this id
            diff += [mv_hat[tar-1:tar]] # motion needed for this id
            target += [x_tar[tar-1:tar]]
        if ref:
            ref = torch.cat(ref,dim=0)
            diff = torch.cat(diff,dim=0)
            target_frames = torch.cat(target,dim=0)
            MC_frames,warped_frames = motioncompensation(warpnet, ref, diff)
            res_tensors = target_frames - MC_frames
            res_hat,_, _,res_act,res_est,_,_ = res_codec(res_tensors)
            com_frames = torch.clip(res_hat + MC_frames, min=0, max=1)
            for i,tar in enumerate(layer):
                if tar>bs:continue
                MC_frame_list[tar-1] = MC_frames[i:i+1]
                warped_frame_list[tar-1] = warped_frames[i:i+1]
                com_frame_list[tar-1] = com_frames[i:i+1]
                res_act_list[tar-1] = res_act[i:i+1]
                res_est_list[tar-1] = res_est[i:i+1]
    MC_frames = torch.cat(MC_frame_list,dim=0)
    warped_frames = torch.cat(warped_frame_list,dim=0)
    com_frames = torch.cat(com_frame_list,dim=0)
    res_est = torch.cat(res_est_list,dim=0)
    res_act = torch.cat(res_act_list,dim=0)
    res_aux = res_codec.entropy_bottleneck.loss()
    return com_frames,MC_frames,warped_frames,res_act,res_est,res_aux
                  
def TreeFrameReconCompress(warpnet,res_codec,x,bs,mv_hat,layers,parents):
    x_tar = x[1:]
    com_frame_list = [None for _ in range(bs)]
    res_act_list = [None for _ in range(bs)]
    x_string_list = []
    z_string_list = []
    for layer in layers:
        ref = [] # reference frame
        diff = [] # motion
        target = [] # target frames
        for tar in layer: # id of frames in this layer
            if tar>bs:continue
            parent = parents[tar]
            ref += [x[:1] if parent==0 else com_frame_list[parent-1]] # ref needed for this id
            diff += [mv_hat[tar-1:tar]] # motion needed for this id
            target += [x_tar[tar-1:tar]]
        if ref:
            ref = torch.cat(ref,dim=0)
            diff = torch.cat(diff,dim=0)
            target_frames = torch.cat(target,dim=0)
            MC_frames,warped_frames = motioncompensation(warpnet, ref, diff)
            res_tensors = target_frames - MC_frames
            res_hat,res_string,_,_,res_act,res_size,_ = res_codec.compress(res_tensors,decodeLatent=True)
            x_string_list += res_string[0]
            z_string_list += res_string[1]
            com_frames = torch.clip(res_hat + MC_frames, min=0, max=1)
            for i,tar in enumerate(layer):
                if tar>bs:continue
                com_frame_list[tar-1] = com_frames[i:i+1]
                res_act_list[tar-1] = res_act[i:i+1]
    res_act = torch.cat(res_act_list,dim=0)
    return (x_string_list,z_string_list),res_act
                              
def TreeFrameReconDecompress(warpnet,res_codec,x_ref,res_string,bs,mv_hat,layers,parents):
    com_frame_list = [None for _ in range(bs)]
    x_string_list,z_string_list = res_string
    for layer in layers:
        ref = [] # reference frame
        diff = [] # motion
        for tar in layer: # id of frames in this layer
            if tar>bs:continue
            parent = parents[tar]
            ref += [x_ref if parent==0 else com_frame_list[parent-1]] # ref needed for this id
            diff += [mv_hat[tar-1:tar]] # motion needed for this id
        if ref:
            ref = torch.cat(ref,dim=0)
            diff = torch.cat(diff,dim=0)
            MC_frames,warped_frames = motioncompensation(warpnet, ref, diff)
            res_string = ([x_string_list.pop(0)],[z_string_list.pop(0)])
            latent_size = torch.Size([ref.size(0),16,16])
            res_hat,_,_,_ = res_codec.decompress(res_string, latentSize=latent_size)
            com_frames = torch.clip(res_hat + MC_frames, min=0, max=1)
            for i,tar in enumerate(layer):
                if tar>bs:continue
                com_frame_list[tar-1] = com_frames[i:i+1]
    com_frames = torch.cat(com_frame_list,dim=0)
    return com_frames


from DVC.subnet import Analysis_mv_net,Synthesis_mv_net,Analysis_prior_net,Synthesis_prior_net,Analysis_net,Synthesis_net,BitEstimator,out_channel_M,out_channel_N,out_channel_mv

class LSVC(nn.Module):
    def __init__(self, name, loss_type='P', compression_level=3, use_split=True):
        super(LSVC, self).__init__()
        self.name = name
        self.useAttn = True if '-A' in name else False
        self.opticFlow = ME_Spynet()
        self.Q = None
        useAttn = ('-A' in name)
        useSynAttn = ('-S' in name)
        channels = 128 if '-128' in name else out_channel_M 
        self.mvEncoder = Analysis_mv_net(useAttn=useAttn,conv_channels=channels)
        self.mvDecoder = Synthesis_mv_net(in_channels=channels,useAttn=useSynAttn)
        self.resEncoder = Analysis_net(useAttn=useAttn)
        self.resDecoder = Synthesis_net(useAttn=useSynAttn)
        self.respriorEncoder = Analysis_prior_net(useAttn=useAttn)
        self.respriorDecoder = Synthesis_prior_net(useAttn=useSynAttn)
        self.bitEstimator_mv = BitEstimator(channels)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.warpnet = Warp_net()
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False
        self.loss_type=loss_type
        self.channels = channels
        self.compression_level=compression_level
        init_training_params(self)
        self.use_split = use_split

    def parallel(self):
        # self.opticFlow = torch.nn.parallel.DistributedDataParallel(self.opticFlow)
        # self.mvDecoder = torch.nn.parallel.DistributedDataParallel(self.mvDecoder)
        # self.resDecoder = torch.nn.parallel.DistributedDataParallel(self.resDecoder)
        # self.respriorDecoder = torch.nn.parallel.DistributedDataParallel(self.respriorDecoder)
        # self.warpnet = torch.nn.parallel.DistributedDataParallel(self.warpnet)
        # self.bitEstimator_z = torch.nn.parallel.DistributedDataParallel(self.bitEstimator_z)
        # self.bitEstimator_mv = torch.nn.parallel.DistributedDataParallel(self.bitEstimator_mv)

        self.opticFlow = torch.nn.DataParallel(self.opticFlow).cuda()
        self.mvDecoder = torch.nn.DataParallel(self.mvDecoder).cuda()
        self.resDecoder = torch.nn.DataParallel(self.resDecoder).cuda()
        self.respriorDecoder = torch.nn.DataParallel(self.respriorDecoder).cuda()
        self.warpnet = torch.nn.DataParallel(self.warpnet).cuda()
        self.bitEstimator_z = torch.nn.DataParallel(self.bitEstimator_z).cuda()
        self.bitEstimator_mv = torch.nn.DataParallel(self.bitEstimator_mv).cuda()
        self.mvEncoder = self.mvEncoder.cuda()
        self.resEncoder = self.resEncoder.cuda()
        self.respriorEncoder = self.respriorEncoder.cuda()

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        if self.use_split:
            inputfeature = inputfeature.cuda(1)
        a = self.warpnet(inputfeature)
        if self.use_split:
            a = a.cuda(0)
        prediction = a + warpframe
        return prediction, warpframe

    def feature_probs_based_sigma(self,feature, sigma):
            
        def getrealbitsg(x, gaussian):
            # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
            cdfs = []
            x = x + self.mxrange
            n,c,h,w = x.shape
            for i in range(-self.mxrange, self.mxrange):
                cdfs.append(gaussian.cdf(i - 0.5).view(n,c,h,w,1))
            cdfs = torch.cat(cdfs, 4).cpu().detach()
            
            byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

            real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

            sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

            return sym_out - self.mxrange, real_bits

        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        
        if self.calrealbits and not self.training:
            decodedx, real_bits = getrealbitsg(feature, gaussian)
            total_bits = real_bits

        return total_bits, probs

    def iclr18_estrate_bits_z(self,z):
        
        def getrealbits(x):
            cdfs = []
            x = x + self.mxrange
            n,c,h,w = x.shape
            for i in range(-self.mxrange, self.mxrange):
                cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(n, 1, h, w, 1))
            cdfs = torch.cat(cdfs, 4).cpu().detach()
            byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

            real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

            sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

            return sym_out - self.mxrange, real_bits

        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


        if self.calrealbits and not self.training:
            decodedx, real_bits = getrealbits(z)
            total_bits = real_bits

        return total_bits, prob


    def iclr18_estrate_bits_mv(self,mv):

        def getrealbits(x):
            cdfs = []
            x = x + self.mxrange
            n,c,h,w = x.shape
            for i in range(-self.mxrange, self.mxrange):
                cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(n, 1, h, w, 1))
            cdfs = torch.cat(cdfs, 4).cpu().detach()
            byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

            real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

            sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
            return sym_out - self.mxrange, real_bits

        prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


        if self.calrealbits and not self.training:
            decodedx, real_bits = getrealbits(mv)
            total_bits = real_bits

        return total_bits, prob

    def res_codec(self,input_residual):
        if self.use_split:
            input_residual = input_residual.cuda(1)
        feature = self.resEncoder(input_residual)
        z = self.respriorEncoder(feature)

        if self.training:
            half = float(0.5)
            noise = torch.empty_like(z).uniform_(-half, half)
            compressed_z = z + noise
        else:
            compressed_z = torch.round(z)

        recon_sigma = self.respriorDecoder(compressed_z)

        feature_renorm = feature

        if self.training:
            half = float(0.5)
            noise = torch.empty_like(feature_renorm).uniform_(-half, half)
            compressed_feature_renorm = feature_renorm + noise
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        recon_res = self.resDecoder(compressed_feature_renorm)

        total_bits_feature, _ = self.feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)
        total_bits = total_bits_feature+total_bits_z
        if self.use_split:
            recon_res = recon_res.cuda(0)
            total_bits = total_bits.cuda(0)
        return recon_res,total_bits

    def mv_codec(self, estmv):
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            half = float(0.5)
            noise = torch.empty_like(mvfeature).uniform_(-half, half)
            quant_mv = mvfeature + noise
        else:
            quant_mv = torch.round(mvfeature)
        quant_mv_upsample = self.mvDecoder(quant_mv)
        total_bits, _ = self.iclr18_estrate_bits_mv(quant_mv)
        return quant_mv_upsample,total_bits

    def forward(self, x):
        input_image = x[1:]
        bs,c,h,w = input_image.size()

        g,layers,parents = graph_from_batch(bs,isLinear=('-L' in self.name),isOnehop=('-O' in self.name))
        ref_index = refidx_from_graph(g,bs)
        t0_enc = t0_dec = time.perf_counter()
        estmv = self.opticFlow(input_image, x[ref_index])
        quant_mv_upsample,total_bits_mv = self.mv_codec(estmv)

        # tree compensation
        MC_frame_list = [None for _ in range(bs)]
        warped_frame_list = [None for _ in range(bs)]
        com_frame_list = [None for _ in range(bs)]
        total_bits_res = None
        x_tar = x[1:]
        for layer in layers:
            ref = [] # reference frame
            diff = [] # motion
            target = [] # target frames
            for tar in layer: # id of frames in this layer
                if tar>bs:continue
                parent = parents[tar]
                ref += [x[:1] if parent==0 else com_frame_list[parent-1]] # ref needed for this id
                diff += [quant_mv_upsample[tar-1:tar]] # motion needed for this id
                target += [x_tar[tar-1:tar]]
            if ref:
                ref = torch.cat(ref,dim=0)
                # if linear or detach, detach it. dont care one-hop
                if '-D' in self.name:
                    ref = ref.detach()
                diff = torch.cat(diff,dim=0)
                target_frames = torch.cat(target,dim=0)
                MC_frames,warped_frames = self.motioncompensation(ref, diff)
                #print(PSNR(target_frames, MC_frames, use_list=True))
                approx_frames = MC_frames
                res_tensors = target_frames - approx_frames
                res_hat,res_bits = self.res_codec(res_tensors)
                com_frames = torch.clip(res_hat + approx_frames, min=0, max=1)
                for i,tar in enumerate(layer):
                    if tar>bs:continue
                    MC_frame_list[tar-1] = MC_frames[i:i+1]
                    warped_frame_list[tar-1] = warped_frames[i:i+1]
                    com_frame_list[tar-1] = com_frames[i:i+1]
                if total_bits_res is None:
                    total_bits_res = res_bits
                else:
                    total_bits_res += res_bits

        self.encoding_time = time.perf_counter() - t0_enc
        self.decoding_time = time.perf_counter() - t0_dec
        MC_frames = torch.cat(MC_frame_list,dim=0)
        warped_frames = torch.cat(warped_frame_list,dim=0)
        com_frames = torch.cat(com_frame_list,dim=0)

        rec_loss = torch.mean((com_frames - input_image).pow(2))
        warp_loss = torch.mean((warped_frames - input_image).pow(2))
        mc_loss = torch.mean((MC_frames - input_image).pow(2))
        
        bpp_res = total_bits_res / (bs * h * w)
        bpp_mv = total_bits_mv / (bs * h * w)
        if self.stage == 'MC' or self.stage == 'WP': bpp_res = bpp_res.detach()
        bpp = (bpp_res + bpp_mv) * self.r_bpp
        
        return com_frames, MC_frames, warped_frames, rec_loss, warp_loss, mc_loss, bpp_res, bpp
       
        
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_DVC_pretrained(level):
    from DVC.net import VideoCompressor, load_model
    model = VideoCompressor()
    model.name = 'DVC-pretrained'
    model.compression_level = level
    model.loss_type = 'P'
    ratio_list = [256,512,1024,2048,2048*2,2048*4,2048*8]
    I_lvl_list = [37,32,27,22,17,12,7]
    model.I_level = I_lvl_list[level]
    model.r = ratio_list[level]
    if level<4:
        global_step = load_model(model, f'DVC/snapshot/{ratio_list[level]}.model')
    net = model.cuda()
    return net


# ---------------------------------BASE MODEL--------------------------------------
class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1) 

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, outplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv2 = nn.Conv2d(in_planes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.shortcut = nn.Sequential()  # do nothing
        if stride != 1 or in_planes != outplanes:
            """For CIFAR10 ResNet paper uses option A."""
            self.shortcut = nn.Sequential(
                                nn.Conv2d(in_planes, outplanes,kernel_size=1, stride=stride, bias=False),
                                nn.BatchNorm2d(outplanes))

    def forward(self, x):
        out = x

        # relu-gate and gate-relu is same
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

from DVC.subnet import AttentionLayer
class CodecNet(nn.Module):
    '''
    Compress residual
    '''
    def __init__(self, cfgs):
        super(CodecNet, self).__init__()
        self.blocks = []
        for cfg in cfgs:
            if isinstance(cfg, int):
                conv_type = cfg
            else:
                conv_type, kernel_size, stride, ch1, ch2 = cfg
            if conv_type == 0:
                layer = nn.Conv2d(ch1, ch2, kernel_size, stride=stride, padding=kernel_size//2)
            elif conv_type == 1:
                layer = nn.ConvTranspose2d(ch1, ch2, kernel_size, stride=stride, padding=kernel_size//2, output_padding=1 if stride==2 else 0)
            elif conv_type == 2:
                layer = nn.ReLU(inplace=True)
            elif conv_type == 3:
                layer = nn.LeakyReLU(negative_slope=0.1)
            # elif conv_type == 4:
            #     layer = GDN(lastCh,inverse=False)
            # elif conv_type == 5:
            #     layer = GDN(lastCh,inverse=True)
            # elif conv_type == 6:
            #     layer = nn.BatchNorm2d(lastCh)
            elif conv_type == 7:
                layer = nn.Tanh()
            elif conv_type == 8:
                layer = BasicBlock(ch1, ch2, stride)
            elif conv_type == 9:
                layer = View()
            elif conv_type == 10:
                layer = nn.AvgPool2d(kernel_size, stride=stride)
            elif conv_type == 11:
                assert ch1==ch2
                layer = AttentionBlock(ch2)
            elif conv_type == 12:
                assert ch1==ch2
                layer = AttentionLayer(ch2)
            elif conv_type == 13:
                layer = ResidualBlock(ch1, ch2)
            else:
                print('conv type not found')
                exit(0)
            self.blocks.append(layer)
        self.blocks = nn.Sequential(*self.blocks)
        for module in self.modules():
            if isinstance(module,nn.Conv2d) or isinstance(module,nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(module.weight.data, math.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias.data, 0.01)

    def forward(self, x,):
        return self.blocks(x)
# STE/norm not so useful
# EC effective, sigmoid or not?
# SSF?
class Base(nn.Module):
    def __init__(self,name,loss_type='P',compression_level=0):
        super(Base, self).__init__()
        self.useEC = True if '-EC' in name else False # sigmoid + concat ===current best===0.061,28.8
        self.useER = True if '-ER' in name else False # predictive quantization mitigates noise
        self.opticFlow = ME_Spynet()
        self.mvEncoder = Analysis_mv_net(out_channels = out_channel_mv)
        self.mvDecoder = Synthesis_mv_net()
        self.warpnet = Warp_net()
        self.bitEstimator_mv = BitEstimator(out_channel_mv)
        self.resEncoder = Analysis_net(out_channels = out_channel_M)

        if self.useEC:
            self.resDecoder = Synthesis_net(in_channels = out_channel_M*2)
        else:
            self.resDecoder = Synthesis_net(in_channels = out_channel_M)
        self.respriorEncoder = Analysis_prior_net(conv_channels = out_channel_N)

        if self.useEC:
            self.respriorDecoder = Synthesis_prior_net(out_channels=out_channel_M*2)
        else:
            self.respriorDecoder = Synthesis_prior_net()

        # error modeling
        if self.useER: 
            ch1,ch2,ch3 = 192,128,128
            kernel_size = 5 # test later
            num_blocks = 1 # better just one is sufficient
            act_func = 4 # 4 (better) or 3 (stable)
            self.residualER = True # must
            self.additiveER = False # both work
            self.detachMode = [0,1] # [0,1] both are better
            self.soft2hard = True;self.s2h_stage = 0
            # ER0 soft
            # possible solution: additive/or not, detachmode=[1], network below, lrelu
            # GDN is better, small kernel=3 may also work, LReLu not good, no additive better, attn not improve
            # GDN good with EREC; LReLu good with ER
            self.mvGenNet = nn.ModuleList([CodecNet(        [(0,kernel_size,1,128,ch1),act_func,(0,kernel_size,1,ch1,ch1),act_func,(0,kernel_size,1,ch1,ch1),act_func,(0,kernel_size,1,ch1,128),act_func]) for _ in range(num_blocks)]) 
            self.resGenNet = nn.ModuleList([CodecNet(       [(0,kernel_size,1,96,ch2),act_func,(0,kernel_size,1,ch2,ch2),act_func,(0,kernel_size,1,ch2,ch2),act_func,(0,kernel_size,1,ch2,96),act_func]) for _ in range(num_blocks)])
            self.respriorGenNet = nn.ModuleList([CodecNet(  [(0,kernel_size,1,64,ch3),act_func,(0,kernel_size,1,ch3,ch3),act_func,(0,kernel_size,1,ch3,ch3),act_func,(0,kernel_size,1,ch3,64),act_func]) for _ in range(num_blocks)])
                
            print(kernel_size,num_blocks, act_func,self.residualER,self.additiveER,self.detachMode,self.soft2hard,self.s2h_stage)
            

        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.warp_weight = 0
        self.mxrange = 150
        self.calrealbits = False
        self.name = name
        self.compression_level = compression_level
        self.loss_type = loss_type
        init_training_params(self)

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe):
        # motion
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        if self.training:
            half = float(0.5)
            quant_noise_mv = torch.empty_like(mvfeature).uniform_(-half, half)
            quant_mv = mvfeature + quant_noise_mv
        else:
            quant_mv = torch.round(mvfeature)
        mv_Q_err = ((mvfeature - torch.round(mvfeature)))

        # we can predict error from feature
        # so enhance the input to decoder
        if self.useER:
            pred_mv = torch.round(mvfeature)
            pred_err_mv = []
            for l in self.mvGenNet:
                if self.residualER:
                    pred_mv = l(pred_mv) + pred_mv
                else:
                    pred_mv = l(pred_mv)
                pred_err_mv += [pred_mv - (mvfeature.detach() if 0 in self.detachMode else mvfeature)]
            corrected_mv = mvfeature + (pred_err_mv[-1].detach() if 1 in self.detachMode else pred_err_mv[-1])
        
        if self.useER:
            if self.training and self.soft2hard and self.s2h_stage > 0:
                quant_mv_upsample = self.mvDecoder(torch.round(mvfeature))
            else:
                quant_mv_upsample = self.mvDecoder(corrected_mv)
        else:
            quant_mv_upsample = self.mvDecoder(quant_mv)

        prediction, warpframe = self.motioncompensation(referframe, quant_mv_upsample)
        if self.training and self.useER and self.soft2hard and self.s2h_stage > 1:
            prediction = prediction.detach()

        # residual   
        input_residual = input_image - prediction
        feature = self.resEncoder(input_residual)
        # quantization
        if self.training:
            half = float(0.5)
            quant_noise_feature = torch.empty_like(feature).uniform_(-half, half)
            compressed_feature_renorm = feature + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature)
        res_Q_err = ((feature - torch.round(feature)))
        
        # hyperprior
        z = self.respriorEncoder(feature)
        # quantization
        if self.training:
            half = float(0.5)
            quant_noise_z = torch.empty_like(z).uniform_(-half, half)
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        z_Q_err = ((z - torch.round(z)))

        if self.useER:
            pred_z = torch.round(z)
            pred_err_z = []
            for l in self.respriorGenNet:
                if self.residualER:
                    pred_z = l(pred_z) + pred_z
                else:
                    pred_z = l(pred_z)
                pred_err_z += [pred_z - (z.detach() if 0 in self.detachMode else z)]
            corrected_z = z + (pred_err_z[-1].detach() if 1 in self.detachMode else pred_err_z[-1])
        
        # rec. hyperprior
        if self.useER:
            if self.training and self.soft2hard and self.s2h_stage > 1:
                recon_sigma = self.respriorDecoder(torch.round(z))
            else:
                recon_sigma = self.respriorDecoder(corrected_z)
        else:
            recon_sigma = self.respriorDecoder(compressed_z)
        if self.useEC:
            recon_sigma, feature_correction = recon_sigma.chunk(2, dim=1)
            feature_correction = torch.sigmoid(feature_correction) - 0.5
            
        # rec. residual
        if self.useER:
            pred_feature = torch.round(feature)
            pred_err_feature = []
            for l in self.resGenNet:
                if self.residualER:
                    pred_feature = l(pred_feature) + pred_feature
                else:
                    pred_feature = l(pred_feature)
                pred_err_feature += [pred_feature - (feature.detach() if 0 in self.detachMode else feature)]
            corrected_feature_renorm = feature + (pred_err_feature[-1].detach() if 1 in self.detachMode else pred_err_feature[-1])
        
        if self.useER:
            if self.training and self.soft2hard and self.s2h_stage > 1:
                resDecInput = torch.round(feature)
            else:
                resDecInput = corrected_feature_renorm 
        else:
            resDecInput = compressed_feature_renorm 
        if self.useEC:
            recon_res = self.resDecoder(torch.cat((resDecInput, feature_correction), dim=1))
        else:
            recon_res = self.resDecoder(resDecInput)

        recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)

        mse_loss = torch.mean((recon_image - input_image).pow(2))
        interloss = torch.mean((prediction - input_image).pow(2))

        def feature_probs_based_sigma(feature, sigma):
            
            def getrealbitsg(x, gaussian):
                # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(gaussian.cdf(torch.tensor(i - 0.5).cuda()).view(n,c,h,w,1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits


            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-5, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            
            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbitsg(feature, gaussian)
                total_bits = real_bits

            return total_bits, probs

        def iclr18_estrate_bits_z(z):
            
            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)

                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(z)
                total_bits = real_bits

            return total_bits, prob


        def iclr18_estrate_bits_mv(mv):

            def getrealbits(x):
                cdfs = []
                x = x + self.mxrange
                n,c,h,w = x.shape
                for i in range(-self.mxrange, self.mxrange):
                    cdfs.append(self.bitEstimator_mv(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
                cdfs = torch.cat(cdfs, 4).cpu().detach()
                byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16), check_input_bounds=True)

                real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())

                sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
                return sym_out - self.mxrange, real_bits

            prob = self.bitEstimator_mv(mv + 0.5) - self.bitEstimator_mv(mv - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))


            if self.calrealbits and not self.training:
                decodedx, real_bits = getrealbits(mv)
                total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estrate_bits_z(compressed_z)
        total_bits_mv, _ = iclr18_estrate_bits_mv(quant_mv)

        im_shape = input_image.size()

        bpp_feature = total_bits_feature / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp_mv = total_bits_mv / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z + bpp_mv
        Q_err = (mv_Q_err).abs().mean() + (res_Q_err).abs().mean() + (z_Q_err).abs().mean()
        Q_std = (mv_Q_err).std() + (res_Q_err).std() + (z_Q_err).std()

        pred_err = 0
        pred_std = 0
        if self.useER:
            for pred_err_x in [pred_err_mv,pred_err_feature,pred_err_z]:
                if self.additiveER:
                    for pe in pred_err_x:
                        pred_err += pe.abs().mean()
                        pred_std += pe.std()
                else:
                    pred_err += pred_err_x[-1].abs().mean()
                    pred_std += pred_err_x[-1].std()
                pred_err /= len(pred_err_x)
                pred_std /= len(pred_err_x)
        
        return clipped_recon_image, mse_loss, interloss, bpp_feature, bpp_z, bpp_mv, bpp, (Q_err, pred_err, Q_std, pred_std)


# utils for scale-space flow
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

from compressai.models.video import ScaleSpaceFlow
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste

class ChannelNorm(nn.Module):
    def __init__(self, channel):
        super(ChannelNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(channel)
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.layer_norm(x)
        x = x.permute(0,3,1,2)
        return x

from super_precision import SPnet
class ELFVC(ScaleSpaceFlow):
    def __init__(
        self,
        name: str,
        loss_type: str='P',
        compression_level: int = 0,
        num_levels: int = 5,
        sigma0: float = 1.5,
        scale_field_shift: float = 1.0,
    ):
        super().__init__(num_levels,sigma0,scale_field_shift)
        class Encoder(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )
        class Decoder(nn.Sequential):
            def __init__(
                self, out_planes: int, in_planes: int = 192, mid_planes: int = 128
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )
        class FlowPredictor(nn.Sequential):
            def __init__(
                self, in_planes: int, mid_planes: int = 128, out_planes: int = 3
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=1),
                )
        class HyperEncoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    conv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(mid_planes, out_planes, kernel_size=5, stride=2),
                )
        class HyperDecoder(nn.Sequential):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__(
                    deconv(in_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, mid_planes, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(mid_planes, out_planes, kernel_size=5, stride=2),
                )
        class HyperDecoderWithQReLU(nn.Module):
            def __init__(
                self, in_planes: int = 192, mid_planes: int = 192, out_planes: int = 192
            ):
                super().__init__()
                from compressai.layers import QReLU
                def qrelu(input, bit_depth=8, beta=100):
                    return QReLU.apply(input, bit_depth, beta)
                self.deconv1 = deconv(in_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu1 = qrelu
                self.deconv2 = deconv(mid_planes, mid_planes, kernel_size=5, stride=2)
                self.qrelu2 = qrelu
                self.deconv3 = deconv(mid_planes, out_planes, kernel_size=5, stride=2)
                self.qrelu3 = qrelu

            def forward(self, x):
                x = self.qrelu1(self.deconv1(x))
                x = self.qrelu2(self.deconv2(x))
                x = self.qrelu3(self.deconv3(x))
                return x
        class Hyperprior(CompressionModel):
            def __init__(self, planes: int = 192, mid_planes: int = 192, side_channel_nc: bool = False, pred_nc: bool = False, 
                        sp: bool = False, no_noise: bool = False):
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(planes)
                self.hyper_encoder = HyperEncoder(planes, mid_planes, planes)
                self.hyper_decoder_mean = HyperDecoder(planes, mid_planes, planes)
                self.hyper_decoder_scale = HyperDecoderWithQReLU(planes, mid_planes, planes)
                self.gaussian_conditional = GaussianConditional(None)
                if pred_nc and not side_channel_nc:
                    self.y_predictor = SPnet(input_channels=planes, output_channels=planes)
                elif not pred_nc and side_channel_nc:
                    self.y_predictor = HyperDecoder(planes, mid_planes, planes)
                elif pred_nc and side_channel_nc:
                    # new
                    self.y_predictor = SPnet(input_channels=planes + 3, output_channels=planes)
                    r = 8
                    self.upsampler = nn.PixelShuffle(r)
                else:
                    self.y_predictor = None
                self.side_channel_nc = side_channel_nc
                self.pred_nc = pred_nc
                self.sp = sp
                self.no_noise = no_noise

            def forward(self, y):
                pred_loss_y = None
                z = self.hyper_encoder(y)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)

                scales = self.hyper_decoder_scale(z_hat)
                means = self.hyper_decoder_mean(z_hat)
                _, y_likelihoods = self.gaussian_conditional(y, scales, means)
                y_hat = quantize_ste(y - means) + means
                Q_err_y = (torch.round(y - means) + means) - y
                pred_err_y = None
                if self.pred_nc and self.side_channel_nc:
                    round_y = torch.round(y - means)
                    side_info = self.upsampler(torch.round(z))
                    all_info = torch.cat((round_y, side_info), dim=1)
                    pred_y = self.y_predictor(all_info) + round_y
                    pred_err_y = pred_y - (y - means).detach()
                    if self.no_noise:
                        # shrinked noise
                        if self.training:
                            # # shift distribution to new
                            # inject_mask = torch.empty_like(y).uniform_(0, float(1)) > 0.5
                            # epsilon = (inject_mask==1) * pred_err_y.detach() + (inject_mask==0) * Q_err_y.detach()
                            # y_hat = y + epsilon
                            pass
                        else:
                            y_hat = pred_y.detach() + means
                    elif self.sp:
                        y_hat = pred_y.detach() + means
                    
                return y_hat, {"y": y_likelihoods, "z": z_likelihoods, "pred_err_y": pred_err_y, "Q_err_y": Q_err_y}
        self.flow_predictor = FlowPredictor(9)
        self.side_channel_nc = True if '-EC' in name else False # sigmoid + concat ===current best===0.061,28.8
        # cat input seems better
        self.pred_nc = True if '-ER' in name else False # predictive quantization mitigates noise, power of 2 or tanh good
        # use single loss is better
        self.compression_level = compression_level
        self.loss_type = loss_type
        init_training_params(self)
        self.spstage = -1
        motion_sp = self.spstage > 0
        res_sp = self.spstage > 1
        motion_nn = '-NN' in name
        self.motion_encoder = Encoder(2 * 3)
        self.motion_decoder = Decoder(2 + 1, in_planes=192)
        self.res_encoder = Encoder(3)
        self.res_decoder = Decoder(3, in_planes=384)
        self.res_hyperprior = Hyperprior(side_channel_nc=self.side_channel_nc, pred_nc=self.pred_nc, sp=res_sp)
        self.motion_hyperprior = Hyperprior(side_channel_nc=self.side_channel_nc, pred_nc=self.pred_nc, sp=motion_sp, no_noise=motion_nn)
        self.name = name
        self.motion_info_prior = None
        self.x_ref_ref = None

    def reset(self):
        self.motion_info_prior = None
        self.x_ref_ref = None

    def optim_parameters(self, epoch, learning_rate):
        lr = learning_rate
        # if epoch < 1:
        #     # train optimizer
        #     self.spstage = 0
        # elif epoch < 7:
        #     # train flow noise predictor
        #     self.spstage = 1
        #     lr *= 10**(-(epoch - 1)//2)
        # else:
        #     self.spstage = 2
        #     lr *= 10**(-(epoch - 7)//2)

        if self.spstage == 0:
            parameters = []
            parameters += self.res_hyperprior.y_predictor.parameters()
            parameters += self.motion_hyperprior.y_predictor.parameters()
        elif self.spstage == 1:
            parameters = []
            parameters += self.motion_hyperprior.y_predictor.parameters()
            parameters += self.motion_decoder.parameters()
            parameters += self.res_encoder.parameters()
            parameters += self.res_decoder.parameters()
            parameters += self.res_hyperprior.parameters()
        elif self.spstage == 2:
            parameters = []
            parameters += self.res_hyperprior.y_predictor.parameters()
            parameters += self.res_decoder.parameters()
        else:
            print('Default stage')
            parameters = [p for n, p in self.named_parameters()]
        print('Current stage:',self.spstage, 'learning rate:', lr)
        return parameters, lr

    def forward_inter(self, x_cur, x_ref):
        if self.motion_info_prior is None:
            B,C,H,W = x_cur.size()
            self.motion_info_prior = torch.zeros(B,3,H,W).to(x_cur.device)
            self.x_ref_ref = torch.zeros(B,3,H,W).to(x_cur.device)

        # flow intial prediction
        motion_info_local = self.flow_predictor(torch.cat((x_ref, self.x_ref_ref, self.motion_info_prior), dim=1))
        x_pred_local = self.forward_prediction(x_ref, motion_info_local)

        # encode the motion information
        y_motion = self.motion_encoder(torch.cat((x_cur, x_pred_local), dim=1))
        y_motion_hat, motion_likelihoods = self.motion_hyperprior(y_motion)

        # decode the space-scale flow information
        motion_info_delta = self.motion_decoder(y_motion_hat)
        motion_info = self.motion_info_prior + motion_info_delta
        x_pred = self.forward_prediction(x_ref, motion_info)

        # residual
        x_res = x_cur - x_pred
        y_res = self.res_encoder(x_res)
        y_res_hat, res_likelihoods = self.res_hyperprior(y_res)

        # y_combine
        x_res_hat = self.res_decoder(torch.cat((y_res_hat, y_motion_hat), dim=1))

        # final reconstruction: prediction + residual
        x_rec = x_pred + x_res_hat

        # record
        self.x_ref_ref = x_ref.detach()
        self.motion_info_prior = motion_info.detach()

        pred_err = []
        if self.pred_nc or self.side_channel_nc:
            for likelihoods in [motion_likelihoods, res_likelihoods]:
                if likelihoods['pred_err_y'] is not None:
                    pred_err += [likelihoods['pred_err_y']]
        Q_err = []
        for likelihoods in [motion_likelihoods, res_likelihoods]:
            if likelihoods['Q_err_y'] is not None:
                Q_err += [likelihoods['Q_err_y']]

        return x_rec, {"motion": motion_likelihoods, "residual": res_likelihoods, "pred_err": pred_err, "Q_err": Q_err}
