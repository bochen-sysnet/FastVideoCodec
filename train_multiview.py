from __future__ import print_function
import os
import sys
import time
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.cuda.amp import autocast as autocast

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from models import get_codec_model,parallel_compression, AverageMeter, compress_whole_video
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

from dataset import FrameDataset, MultiViewVideoDataset
import pytorch_msssim

parser = argparse.ArgumentParser(description='PyTorch EAVC Training')
parser.add_argument('--benchmark', action='store_true',
                    help='benchmark model on validation set')
parser.add_argument('--super-batch', default=1, type=int,
                    help="super batch size")
parser.add_argument('--num-views', default=0, type=int,
                    help="number of views")
parser.add_argument('--debug', action='store_true',
                    help='debug model on validation set')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--codec', type=str, default='MCVC-IA',
                    help='name of codec')
parser.add_argument('--device', default=0, type=int,
                    help="GPU ID")
parser.add_argument('--epoch', type=int, nargs='+', default=[0,1000],
                    help='Begin and end epoch. Not useful in this exp since the total epoch is determined by computation power.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--loss-type', type=str, default='P',
                    help='PSNR or MS-SSIM')
parser.add_argument('--resume', type=str, default='',
                    help='Resume path')
parser.add_argument('--resilience', default=10, type=int,
                    help="Number of losing views to tolerate")
parser.add_argument('--force-resilience', default=0, type=int,
                    help="Force the number of losing views in training/evaluation. Fixed to 0 in this exp.")
parser.add_argument("--fP", type=int, default=15, help="The number of forward P frames")
parser.add_argument("--bP", type=int, default=0, help="The number of backward P frames")
parser.add_argument('--level-range', type=int, nargs='+', default=[0,4])
parser.add_argument('--frame-comb', default=0, type=int, help="Frame combination method. 0: naive. 1: spatial. 2: temporal.")
parser.add_argument('--pretrain', action='store_true',
                    help='Pretrain model offline.')
parser.add_argument('--onlydecoder', action='store_true',
                    help='only train decoder enhancement part')
parser.add_argument('--data-ratio', type=float, default=1,
                    help='The ratio of dataset in training')
parser.add_argument('--sample-interval', type=int, default=1,
                    help='The ratio of frame sampling in streaming')
parser.add_argument('--c2s-ratio', type=float, default=1.33,
                    help='The ratio of computation to streaming speed [1.33,0.87,0.7]')
parser.add_argument('--sample-ratio', type=float, default=1,
                    help='The ratio of sampled pixels in streaming in streaming')
parser.add_argument('--compression-level', default=0, type=int,
                    help="Compression level: 0,1,2,3")
parser.add_argument('--category-id', default=0, type=int,
                    help="Category I: 0,1,2,3,4")

args = parser.parse_args()

# OPTION
CODEC_NAME = args.codec
SAVE_DIR = f'backup/{CODEC_NAME}'
loss_type = args.loss_type
LEARNING_RATE = args.lr
WEIGHT_DECAY = 5e-4
device = args.device
STEPS = [1]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

seed = int(time.time())
# seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

def get_model_n_optimizer_n_score_from_level(codec_name,compression_level,category_id,pretrain=False,
                                             onlydecoder=False,num_views=0,use_attn=True,load_pretrain=True):
    # codec model
    model = get_codec_model(codec_name, 
                            loss_type=loss_type, 
                            compression_level=compression_level,
                            use_split=False,
                            num_views=num_views if not pretrain else 1,
                            resilience=args.resilience,
                            use_attn=use_attn)
    model.force_resilience = args.force_resilience
    model.sample_ratio = args.sample_ratio
    model = model.cuda(device)

    def load_from_path(path):
        # initialize best score
        best_codec_score = 100
        print("Loading for ", codec_name, 'from',path)
        checkpoint = torch.load(path,map_location=torch.device('cuda:'+str(device)))
        load_state_dict_all(model, checkpoint['state_dict'])
        # load_state_dict_whatever(model, checkpoint['state_dict'])
        if 'stats' in checkpoint:
            best_codec_score = checkpoint['stats'][0] - checkpoint['stats'][1]
            print("Loaded model codec stat: ", checkpoint['stats'],', score:', best_codec_score)
        del checkpoint
        return best_codec_score

    best_codec_score = 100
    paths = []
    if args.resume:
        paths += [args.resume]
    # training order
    # IA-PT, IA0 (no fault-tolerance), IA (with fault-tolerance)
    if 'MCVC-IA-OLFT' in codec_name and load_pretrain:
        paths += [f'backup/MCVC-IA-PT/MCVC-IA-PT-{compression_level}{loss_type}_vid0_best.pth']
        paths += [f'backup/MCVC-IA-PT/MCVC-IA-PT-{compression_level}{loss_type}_vid0_ckpt.pth']
    paths += [f'{SAVE_DIR}/{codec_name}-{compression_level}{loss_type}_vid{category_id}_best.pth']
    paths += [f'{SAVE_DIR}/{codec_name}-{compression_level}{loss_type}_vid{category_id}_ckpt.pth']
    for pth in paths:
        if os.path.isfile(pth):
            best_codec_score = load_from_path(pth)
            break

    # create optimizer
    if not onlydecoder:
        parameters = [p for n, p in model.named_parameters()]
    else:
        parameters = [p for n, p in model.named_parameters() if 'backup' in n]
    optimizer = torch.optim.Adam([{'params': parameters}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pytorch_total_params = sum(p.numel() for p in parameters)
    print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    return model, optimizer, best_codec_score


def metrics_per_gop(out_dec, raw_frames, ssim=False, training=False):
    frame_idx = 0
    total_bpp = 0
    total_psnr = 0
    total_mse = 0
    pixels = 0
    completeness = 1
    non_zero_indices = out_dec['non_zero_indices'] if 'non_zero_indices' in out_dec else None
    if non_zero_indices is not None:
        completeness = 1.0 * len(non_zero_indices) / raw_frames[0].size(0)
    for x_hat,likelihoods in zip(out_dec['x_hat'],out_dec['likelihoods']):
        if training and 'OLFT' in args.codec:
            x = out_dec['x_touch'][frame_idx]
        else:
            x = raw_frames[frame_idx]
        for likelihood_name in ['keyframe', 'motion', 'residual']:
            if likelihood_name in likelihoods:
                var_like = likelihoods[likelihood_name]
                bits = torch.sum(torch.clamp(-1.0 * torch.log(var_like["y"] + 1e-5) / math.log(2.0), 0, 50)) + \
                        torch.sum(torch.clamp(-1.0 * torch.log(var_like["z"] + 1e-5) / math.log(2.0), 0, 50))
    
        if ssim:
            if non_zero_indices is None:
                mseloss = 1 - pytorch_msssim.ms_ssim(x_hat, x)
            else:
                mseloss = 1 - pytorch_msssim.ms_ssim(x_hat[non_zero_indices], x[non_zero_indices])
        else:
            if non_zero_indices is None:
                mseloss = torch.mean((x_hat - x).pow(2))
            else:
                mseloss = torch.mean((x_hat[non_zero_indices] - x[non_zero_indices]).pow(2))
        psnr = 10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(raw_frames[0].device)


        # if use touch-ups training
        if training and 'OLFT' in args.codec:
            total_bpp += out_dec['x_touch_bits'][frame_idx] / bits
        else:
            # supervise the ref frame
            if 'x_ref' in out_dec and out_dec['x_ref']:
                mseloss += torch.mean((out_dec['x_ref'][frame_idx] - x).pow(2))
                mseloss /= 2
            # calc bpp
            pixels = x.size(0) * x.size(2) * x.size(3)
            total_bpp += bits / pixels
        total_psnr += psnr
        total_mse += mseloss
        frame_idx += 1

    return total_mse/frame_idx,total_bpp/frame_idx,total_psnr/frame_idx,completeness
        
def train(epoch, model, train_dataset, optimizer, pretrain=False, probe=False, print_header=None, codec=''):
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    ssim_module = AverageMeter()
    psnr_module = AverageMeter()
    all_loss_module = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ds_size = len(train_dataset)
    
    model.train()
    # multi-view dataset must be single batch in loader 
    # single view dataset set batch size to view numbers in loader in test
    batch_size = 8 if pretrain else 1
    num_workers = batch_size if pretrain else 1 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                               num_workers=num_workers, drop_last=True, pin_memory=True)
    
    train_iter = tqdm(train_loader)
    for batch_idx,data in enumerate(train_iter):
        if not pretrain:
            b,g,v,c,h,w = data.size()
            data = data.permute(1,0,2,3,4,5).reshape(g,b*v,c,h,w).cuda(device)
        else:
            data = data.cuda(device)
        
        # run model
        out_dec = model(data)
        mse, bpp, psnr, completeness = metrics_per_gop(out_dec, data, ssim=False, training=True)
        _, _, ssim, _ = metrics_per_gop(out_dec, data, ssim=True, training=True)
        if 'OLFT' in args.codec:
            loss = model.r*mse
        else:
            loss = model.r*mse + bpp
        
        ba_loss_module.update(bpp.cpu().data.item())
        psnr_module.update(psnr.cpu().data.item())
        ssim_module.update(ssim.cpu().data.item())
        img_loss_module.update(mse.cpu().data.item())
        all_loss_module.update(loss.cpu().data.item())
        
        # backward
        scaler.scale(loss).backward()
        if batch_idx % args.super_batch == args.super_batch-1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # show result
        train_iter.set_description(
            f"{epoch},{args.compression_level}. "
            f"L:{all_loss_module.val:.4f} ({all_loss_module.avg:.4f}). "
            f"I:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"S:{ssim_module.val:.2f} ({ssim_module.avg:.2f}). ")
        
        # return probe result
        if probe and batch_idx>=5:
            return ba_loss_module.avg
        
        if print_header is not None:
            with open(f'{codec}.log','a') as f:
                f.write(f'{print_header[0]},{print_header[1]},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{ssim_module.val:.4f}\n')

    return ba_loss_module.avg-psnr_module.avg, [ba_loss_module.avg,psnr_module.avg,ssim_module.avg]
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    LR_DECAY_RATE = 0.1
    r = (LR_DECAY_RATE ** (sum(epoch >= np.array(STEPS))))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= r
    return r
    
def save_checkpoint(state, is_best, directory, CODEC_NAME, loss_type, compression_level, category_id):
    import shutil
    epoch = state['epoch']; bpp = state['stats'][0]; psnr = state['stats'][1]; score=state['score']
    ckpt_filename = f'{directory}/{CODEC_NAME}-{compression_level}{loss_type}_vid{category_id}_ckpt.pth'
    best_filename = f'{directory}/{CODEC_NAME}-{compression_level}{loss_type}_vid{category_id}_best.pth'
    torch.save(state, ckpt_filename)
    if is_best:
        shutil.copyfile(ckpt_filename, best_filename)
        print('Saved to:',best_filename)

    with open(f'{directory}/log.txt','a+') as f:
        f.write(f'{category_id},{compression_level},{args.data_ratio},{args.sample_ratio},{args.sample_interval},{epoch},{bpp},{psnr},{score}\n')

def test(epoch, model, test_dataset, print_header=None, codec=''):
    model.eval()
    ssim_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    ds_size = len(test_dataset)
    
    data = []
    # test must use a batch size equal to the number of views/cameras
    test_iter = tqdm(range(ds_size))
    eof = False
    for data_idx,_ in enumerate(test_iter):
        data = test_dataset[data_idx].cuda(device)
        if args.codec == 'MCVC-Original':
            data = [data[g] for g in range(data.size(0))]
        with torch.no_grad():
            out_dec = model(data)
            mse, bpp, psnr, completeness = metrics_per_gop(out_dec, data, ssim=False)
            _, _, ssim, _ = metrics_per_gop(out_dec, data, ssim=True)
            
            ba_loss_module.update(bpp.cpu().data.item())
            psnr_module.update(psnr.cpu().data.item())
            ssim_module.update(ssim.cpu().data.item())
                
        # metrics string
        # metrics_str = ""
        # for i,(psnr,bpp) in enumerate(zip(psnr_vs_resilience,bpp_vs_resilience)):
        #     metrics_str += f"{psnr.count}:{psnr.avg:.2f},{bpp.avg:.3f}. "

        # show result
        test_iter.set_description(
            f"{epoch} {data_idx:6}. "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.4f} ({psnr_module.avg:.4f}). "
            f"S:{ssim_module.val:.4f} ({ssim_module.avg:.4f}). ")

        if print_header is not None:
            with open(f'{codec}.log','a') as f:
                f.write(f'{print_header[0]},{print_header[1]},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{ssim_module.val:.4f}\n')
        if args.debug and data_idx == 9:exit(0)
    if print_header is not None:
        with open(f'{codec}.avg.log','a') as f:
            f.write(f'{print_header[0]},{print_header[1]},{ba_loss_module.avg:.4f},{psnr_module.avg:.4f},{ssim_module.avg:.4f}\n')
    # test_dataset.reset()        
    return ba_loss_module.avg-psnr_module.avg, [ba_loss_module.avg,psnr_module.avg,ssim_module.avg]

def static_simulation_x26x_multicam(args,test_dataset,category_id):
    ds_size = len(test_dataset)
    quality_levels = [7,11,15,19,23,27,31,35]
    # quality_levels = [15,19,23,27]
    
    Q_list = quality_levels#[args.level_range[0]:args.level_range[1]]
    for lvl,Q in enumerate(Q_list):
        data = []
        ba_loss_module = AverageMeter()
        psnr_module = AverageMeter()
        msssim_module = AverageMeter()
        test_iter = tqdm(range(ds_size))
        for data_idx,_ in enumerate(test_iter):
            data = test_dataset[data_idx]

            l = len(data)
                
            psnr_list,msssim_list,bpp_act_list,compt,decompt = compress_whole_video(args.codec,data,Q,*test_dataset._frame_size, GOP=args.fP + args.bP +1, frame_comb=args.frame_comb)
            
            # aggregate loss
            ba_loss = torch.stack(bpp_act_list,dim=0).mean(dim=0)
            psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
            msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
            
            # record loss
            ba_loss_module.update(ba_loss.cpu().data.item(), l)
            psnr_module.update(psnr.cpu().data.item(),l)
            msssim_module.update(msssim.cpu().data.item(), l)

            # show result
            test_iter.set_description(
                f"Q:{Q}"
                f"{data_idx:6}. "
                f"BA: {ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
                f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
                f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). ")

            # write result
            with open(f'{args.codec}.log','a') as f:
                f.write(f'{category_id},{lvl},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{msssim_module.val:.4f}\n')


        # write result
        with open(f'{args.codec}.avg.log','a') as f:
            f.write(f'{category_id},{lvl},{ba_loss_module.avg:.4f},{psnr_module.avg:.4f},{msssim_module.avg:.4f}\n')

def probe_sample_interval(use_compression=True,probe_dataset=None,probe_model=None,optimizer=None):
    if probe_dataset is None:
        probe_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()]),
            category_id=args.category_id,num_views=args.num_views,
            data_ratio=1, sample_interval=1,c2s_ratio=1)
    if probe_model is None or optimizer is None:
        probe_model, optimizer, _ = get_model_n_optimizer_n_score_from_level(args.codec,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                    num_views=probe_dataset.num_views)
    probe_model.real_replace = True
    probe_model.use_compression = use_compression
    ratio = train(0, probe_model, probe_dataset, optimizer, probe=True)
    probe_model.real_replace = False
    return int(ratio/0.01)

def static_simulation_model_multicam(args, test_dataset,category_id):
    if CODEC_NAME == 'MCVC-Original':
        args.level_range = [0,9]
    for lvl in range(args.level_range[0],args.level_range[1]):
        model, _, _ = get_model_n_optimizer_n_score_from_level(CODEC_NAME,lvl,category_id,num_views=test_dataset.num_views)
        test(0, model, test_dataset, print_header=[category_id,lvl], codec=args.codec)

# test per video
if args.benchmark:
    for category_id in range(5):
        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',transform=shared_transforms,\
            category_id=category_id,num_views=args.num_views)
        if 'x26' in args.codec:
            static_simulation_x26x_multicam(args, test_dataset, category_id)
        else:
            static_simulation_model_multicam(args, test_dataset, category_id)
    exit(0)

if args.evaluate:
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',transform=shared_transforms,category_id=args.category,num_views=args.num_views)

    model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level,args.category,num_views=test_dataset.num_views)
    score, stats = test(0, model, test_dataset)
    exit(0)

# pretraining uses data from generic scenes
if args.pretrain:
    train_dataset = FrameDataset('../dataset/vimeo', frame_size=256) 
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',
                        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views)
    
    model, optimizer, best_pretrain_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                                    0, pretrain=True, onlydecoder=args.onlydecoder,
                                                                                    num_views=test_dataset.num_views)
    cvg_cnt = 0
    BEGIN_EPOCH = args.epoch[0]
    END_EPOCH = args.epoch[1]
    for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
        train(epoch, model, train_dataset, optimizer, pretrain=True)
        score, stats = test(epoch, model, test_dataset)
        is_best = score <= best_pretrain_score
        if is_best:
            print("New best", stats, "Score:", score, ". Previous: ", best_pretrain_score)
            best_pretrain_score = score
            cvg_cnt = 0
        else:
            cvg_cnt += 1
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score, 'stats': stats}
        save_checkpoint(state, is_best, SAVE_DIR, CODEC_NAME, loss_type, args.compression_level, 0)
        if cvg_cnt == 10:break
    exit(0)

assert args.onlydecoder, "Only support decoder update now to accelerate!"

# use category 1, 6 views for example
# test schedule:
# 0. Check it works on different category (done)
# 1. MCVC-IA-OLFT test for every category and level based on MCVC-IA-PT (done)
# 2. MCVC-IA-OLFT test different c2s ratios (1.33,0.87,0.7), same category, all levels (done)
# 3. MCVC-IA-OLFT test different frame sampling interval (1,10,100,1000), same category, all levels (ongoing)
# 4. MCVC-IA-OLFT test different pixel sampling ratios (1,0.1,0.01,0.001), same category, all levels (ongoing)
# 5. MCVC-IA-OLFT test different #view (1,2,3,4,5,6), 6-view category, all levels
# 6. MCVC-IA-OLFT without pre-training vs. without pre-train and attention, same category, all levels
# 7. progress of sample rate



for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    for sr in [0.001,0.01,0.1,1]:
        args.sample_ratio = sr

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("Sampling interval:",si_no_compression,args.sample_interval,"cat:",args.category_id,
              "c2s",args.c2s_ratio,"si",args.sample_interval,"sr",args.sample_ratio)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats = train(0, model, train_dataset, optimizer)

        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.sr.log','a') as f:
            f.write(f'{args.sample_ratio},{args.compression_level},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')

for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    for si in [0,1,10,100,1000]:
        args.sample_interval = si

        print("Sampling interval:",args.sample_interval,"cat:",args.category_id,
              "c2s",args.c2s_ratio,"si",args.sample_interval)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats = train(0, model, train_dataset, optimizer)
        with open(f'MCVC-IA-OLFT.si.log','a') as f:
            f.write(f'{args.sample_interval},{args.compression_level},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')
exit(0)
for cl in range(4):
    args.compression_level = cl
    for cat in range(5):
        args.category_id = cat

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("Sampling interval:",si_no_compression,args.sample_interval,"cat:",args.category_id)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats0 = test(0, model, train_dataset)

        train_dataset.reset()
        _, stats = train(0, model, train_dataset, optimizer, print_header=[args.category_id, args.compression_level], codec=args.codec)
        
        # record stat
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.cat.log','a') as f:
            f.write(f'{args.category_id},{args.compression_level},{stats0[0]:.4f},{stats0[1]:.4f},{stats0[2]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')
exit(0)

for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
        data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

    model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                args.category_id, onlydecoder=args.onlydecoder,
                                                            num_views=train_dataset.num_views,)

    _, stats = train(0, model, train_dataset, optimizer, print_header=[args.category_id, args.compression_level], codec='probe')
exit(0)

for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    for use_attn in [False,True]:

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("Sampling interval:",si_no_compression,args.sample_interval,"cat:",args.category_id,
              "c2s",args.c2s_ratio,"si",args.sample_interval,"sr",args.sample_ratio,args.num_views)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views,use_attn=use_attn,load_pretrain=False)

        _, stats = train(0, model, train_dataset, optimizer)
        
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.ablation.log','a') as f:
            f.write(f'{use_attn},{args.compression_level},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')
exit(0)

for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    for nv in range(1,7):
        args.num_views = nv

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("Sampling interval:",si_no_compression,args.sample_interval,"cat:",args.category_id,
              "c2s",args.c2s_ratio,"si",args.sample_interval,"sr",args.sample_ratio,args.num_views)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats = train(0, model, train_dataset, optimizer)
        
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.nv.log','a') as f:
            f.write(f'{args.num_views},{args.compression_level},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')
exit(0)

for cl in range(4):
    args.compression_level = cl
    args.category_id = 1
    for c2s in [0.7,0.87]:
        args.c2s_ratio = c2s

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("Sampling interval:",si_no_compression,args.sample_interval,"cat:",args.category_id,"c2s",args.c2s_ratio)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats = train(0, model, train_dataset, optimizer)

        # record stat
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.c2s.log','a') as f:
            f.write(f'{args.c2s_ratio},{args.compression_level},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')
exit(0)