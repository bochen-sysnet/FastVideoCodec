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
from models import load_state_dict_all

from dataset import FrameDataset, MultiViewVideoDataset
import pytorch_msssim

parser = argparse.ArgumentParser(description='PyTorch EAVC Training')
parser.add_argument('--benchmark', action='store_true',
                    help='benchmark model on validation set')
parser.add_argument('--simulate', action='store_true',
                    help='simulate model')
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
parser.add_argument('--speed-test', action='store_true',
                    help='Test model speed.')
parser.add_argument('--onlydecoder', action='store_true',
                    help='only train decoder enhancement part')
parser.add_argument('--data-ratio', type=float, default=1,
                    help='The ratio of dataset in training')
parser.add_argument('--sample-interval', type=float, default=1,
                    help='The ratio of frame sampling in streaming')
parser.add_argument('--c2s-ratio', type=float, default=1.33,
                    help='The ratio of computation to streaming speed [1.33,0.87,0.7]')
parser.add_argument('--sample-ratio', type=float, default=1,
                    help='The ratio of sampled pixels in streaming in streaming')
parser.add_argument('--compression-level', default=0, type=int,
                    help="Compression level: 0,1,2,3")
parser.add_argument('--category-id', default=1, type=int,
                    help="Category I: 0,1,2,3,4")
parser.add_argument('--bw-limit', type=float, default=0.01,
                    help='The desired ratio of extra bandwidth')
parser.add_argument('--max-pool-size', type=float, default=10000,
                    help='The ratio of max pool size to num of gops')

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
                                             onlydecoder=False,num_views=0,use_attn=True,load_pretrain=True,
                                             load_with_copy=False,sample_ratio=1):
    # codec model
    model = get_codec_model(codec_name, 
                            loss_type=loss_type, 
                            compression_level=compression_level,
                            use_split=False,
                            num_views=num_views if not pretrain else 1,
                            resilience=args.resilience,
                            use_attn=use_attn,
                            load_with_copy=load_with_copy)
    model.force_resilience = args.force_resilience
    model.sample_ratio = sample_ratio
    model = model.cuda(device)

    def load_from_path(path):
        # initialize best score
        best_codec_score = 100
        print("Loading for ", codec_name, 'from',path)
        checkpoint = torch.load(path,map_location=torch.device('cuda:'+str(device)))
        load_state_dict_all(model, checkpoint['state_dict'])
        if 'stats' in checkpoint:
            best_codec_score = checkpoint['stats'][0] - checkpoint['stats'][1]
            print("Loaded model codec stat: ", checkpoint['stats'],', score:', best_codec_score)
        del checkpoint
        return best_codec_score

    best_codec_score = 100
    paths = []
    if 'MCVC-IA-OLFT' in codec_name and load_pretrain:
        if args.resume:
            paths += [args.resume]
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
        
def train(epoch, model, train_dataset, optimizer, pretrain=False, probe=False, print_header=None, codec='', test_dataset=None):
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
                f.write(f'{print_header[0]},{batch_idx},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{ssim_module.val:.4f}\n')

        if test_dataset is not None and (batch_idx%1000 == 999 or batch_idx == len(train_dataset)):
            _, stats = test(0, model, test_dataset)
            model.train()
            with open(f'progress.log','a') as f:
                f.write(f'{print_header[0]},{batch_idx},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')

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

        if args.debug and data_idx == 9:exit(0)
    # test_dataset.reset()        
    return ba_loss_module.avg-psnr_module.avg, [ba_loss_module.avg,psnr_module.avg,ssim_module.avg]

def static_simulation_x26x_multicam(args,test_dataset,category_id):
    ds_size = len(test_dataset)
    # quality_levels = [7,11,15,19,23,27,31,35]
    quality_levels = [23,27,31,35]
    Q_list = quality_levels
    for lvl,Q in enumerate(Q_list):
        # if category_id == 3 and lvl<=5:
        #     continue
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

            # # write result
            # with open(f'{args.codec}.log','a') as f:
            #     f.write(f'{category_id},{lvl},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{msssim_module.val:.4f}\n')


        # write result
        with open(f'{args.codec}.avg.log','a') as f:
            f.write(f'{category_id},{lvl},{ba_loss_module.avg:.4f},{psnr_module.avg:.4f},{msssim_module.avg:.4f}\n')

def probe_sample_interval(use_compression=True,probe_dataset=None,probe_model=None,optimizer=None,sample_ratio=1):
    if probe_dataset is None:
        probe_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()]),
            category_id=args.category_id,num_views=args.num_views,
            data_ratio=1, sample_interval=1,c2s_ratio=1)
    if probe_model is None or optimizer is None:
        probe_model, optimizer, _ = get_model_n_optimizer_n_score_from_level(args.codec,args.compression_level, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                    num_views=probe_dataset.num_views,sample_ratio=sample_ratio)
    probe_model.real_compression = True
    probe_model.use_compression = use_compression
    ratio = train(0, probe_model, probe_dataset, optimizer, probe=True)
    probe_model.real_compression = False
    return max(1, ratio/args.bw_limit)

def static_simulation_model_multicam(args, test_dataset,category_id):
    if CODEC_NAME == 'MCVC-Original':
        args.level_range = [0,9]
    for lvl in range(args.level_range[0],args.level_range[1]):
        model, _, _ = get_model_n_optimizer_n_score_from_level(CODEC_NAME,lvl,category_id,num_views=test_dataset.num_views)
        _, stats = test(0, model, test_dataset)
        with open(f'{args.codec}.avg.log','a') as f:
            f.write(f'{category_id},{lvl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')

def simulation():
    # read network traces
    import csv
    total_traces = 100
    single_trace_len = 10000
    for trace_id in range(2):
        downthrpt = []
        with open('../curr_videostream.csv', mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                downthrpt_sample = float(row["downthrpt"])*8
                if trace_id == 0:
                    # constraint
                    if downthrpt_sample<10e6:
                        downthrpt += [downthrpt_sample]
                else:
                    downthrpt += [downthrpt_sample]
                if len(downthrpt) >= single_trace_len * total_traces:
                    break
        print('Trace stats:',np.array(downthrpt).mean(),np.array(downthrpt).std(),np.array(downthrpt).max(),np.array(downthrpt).min())
        
        views_of_category = [4,6,5,4,4]
        for codec in ['x264-veryslow', 'x265-veryslow', 'MCVC-Original']:
            filename = f'{codec}.log'
            bpp_arr = [[[] for _ in range(4)] for _ in range(5)]
            psnr_arr = [[[] for _ in range(4)] for _ in range(5)]
            ssim_arr = [[[] for _ in range(4)] for _ in range(5)]
            with open(filename, mode='r') as f:
                line_count = 0
                for l in f.readlines():
                    l = l.split(',')
                    cat,lvl,bpp,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
                    bpp_arr[cat][lvl] += [bpp]
                    psnr_arr[cat][lvl] += [psnr]
                    ssim_arr[cat][lvl] += [ssim]
                    if cat == 0 and lvl == 0:
                        line_count += 1
            print('Loaded codec:',codec)
            sim_iter = tqdm(range(total_traces))
            for _,i in enumerate(sim_iter):
                trace_start = i * single_trace_len
                trace_end = trace_start + single_trace_len
                bw_list = downthrpt[trace_start:trace_end]
                for cat in range(5):
                    bpp_module = AverageMeter()
                    bpp_list = []
                    psnr_module = AverageMeter()
                    ssim_module = AverageMeter()
                    for gop_id in range(line_count):
                        max_bpp = bw_list[gop_id] / views_of_category[cat] / (1920*1080)
                        for lvl in range(4):
                            if bpp_arr[cat][lvl][gop_id] > max_bpp:
                                break
                        bpp_module.update(bpp_arr[cat][lvl][gop_id])
                        bpp_list += [bpp_arr[cat][lvl][gop_id]]
                        psnr_module.update(psnr_arr[cat][lvl][gop_id])
                        ssim_module.update(ssim_arr[cat][lvl][gop_id])
                    sorted_bpp = np.sort(np.array(bpp_list))
                    N = len(bpp_list)
                    bpp99 = sorted_bpp[int(N*1/100)]
                    bpp999 = sorted_bpp[int(N*1/1000)]
                    with open(f'simulation.log', mode='r') as f:
                        f.write(f'{trace_id},{codec},{cat},{bpp_module.avg},{psnr_module.avg},{ssim_module.avg},{bpp99},{bpp999}\n')

if args.simulate:
    simulation()
    exit(0)

# test per video
# measure per video stats for cat 1
if args.benchmark:
    gop_size=250 if 'x26' in args.codec else 16
    for category_id in range(5):
        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',transform=shared_transforms,\
            category_id=category_id,num_views=args.num_views,gop_size=gop_size)
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

if args.speed_test:
    # speed of more views, olft vs. original

    nv = 6 #1...6
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',
        transform=shared_transforms,category_id=args.category_id,num_views=nv,)
    model, _, _ = get_model_n_optimizer_n_score_from_level(CODEC_NAME,0,args.category_id,num_views=test_dataset.num_views)
    _, stats = test(0, model, test_dataset)
# encoder
# original: [5.94,4.30,3.27,2.50,2.05,1.78]
# decoder is a bit complex as there are two
# 0.003514678156313797 0.005066935391165316
# 0.0038382569160312413 0.005380996245269974
# 0.0041302489172667265 0.005700193654124936
# 0.004484572874847799 0.006108296576421708
# 0.004801990651059896 0.006426506982650608
# 0.005413683934137225 0.007108943206723779

assert args.onlydecoder, "Only support decoder update now to accelerate!"
# use category 1, 6 views for example
# test schedule:
# 0. Check it works on different category (done)
# 1. MCVC-IA-OLFT test for every category and level based on MCVC-IA-PT (done)
# 2. MCVC-IA-OLFT test different c2s ratios (1.33,0.87,0.7), same category, all levels (done)
# 3. MCVC-IA-OLFT test different frame sampling interval (1,10,100,1000), same category, all levels (ongoing)
# 4. MCVC-IA-OLFT test different pixel sampling ratios (1,0.1,0.01,0.001), same category, all levels (done)
# 5. MCVC-IA-OLFT test different #view (1,2,3,4,5,6), 6-view category, all levels
# 6. MCVC-IA-OLFT without pre-training vs. without pre-train and attention, same category, all levels
# 7. progress of sample rate
# 8. vary pool max size
# 9. vary data ratio
# 10. train each level until no improvement in three consecutive iterations, record perf each epoch


for cl in range(4):
    args.compression_level = cl
    for cat in range(5):
        args.category_id = cat

        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("SI:",si_no_compression,args.sample_interval,"cat:",cat)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=cat,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=args.c2s_ratio)
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=cat,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level, 
                                                                    cat, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats0 = test(0, model, test_dataset)

        train(0, model, train_dataset, optimizer)


        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=cat,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        _, stats = test(0, model, test_dataset)
        
        # record stat
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.cat.log','a') as f:
            f.write(f'{args.category_id},{args.compression_level},{stats0[0]:.4f},{stats0[1]:.4f},{stats0[2]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')
exit(0)
for cl in [3]:
    for nv in range(1,7):
        converge_cnt = 0
        best_psnr = 0

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='train',
            transform=shared_transforms,category_id=args.category_id,num_views=nv,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',
            transform=shared_transforms,category_id=args.category_id,num_views=nv,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        print("LONG cat:",args.category_id, "c2s",args.c2s_ratio,"sr",args.sample_ratio,'NV:',nv,train_dataset.num_views)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=False,
                                                                num_views=nv)

        for epoch in range(100):
            train(0, model, train_dataset, optimizer)
            _, stats = test(0, model, test_dataset)
        
            with open(f'MCVC-IA-OLFT.longterm.nv.log','a') as f:
                f.write(f'{cl},{nv},{epoch},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')

            if stats[1] > best_psnr:
                converge_cnt = 0
                best_psnr = stats[1]
            else:
                converge_cnt += 1
                if converge_cnt == 3:
                    break

exit(0)


for cl in range(4):
    converge_cnt = 0
    best_psnr = 0
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='train',
        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
        data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)

    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',
        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
        data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
    
    model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                args.category_id, onlydecoder=args.onlydecoder,
                                                            num_views=train_dataset.num_views)

    for epoch in range(100):
        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)

        if stats[2] > best_psnr:
            converge_cnt = 0
            best_psnr = stats[2]
        else:
            converge_cnt += 1
            if converge_cnt == 3:
                break
            
        with open(f'MCVC-IA-OLFT.longterm.log','a') as f:
            f.write(f'{cl},{epoch},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')


for cl in range(4):
    for nv in range(1,7):
        si = probe_sample_interval(True)
        print("SHORT cat:",args.category_id, "c2s",args.c2s_ratio,"si",si,"sr",args.sample_ratio,'NV:',nv)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=nv,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=nv,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=nv)

        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
    
        with open(f'MCVC-IA-OLFT.shortterm.nv.log','a') as f:
            f.write(f'{cl},{nv},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f}\n')

exit(0)
for cl in range(4):
    # for dr in [0.2,0.4,0.6,0.8,1]:
    # for dr in [0.01,0.02,0.03,0.04,0.05,0.1]:
    for dr in [0.06,0.07,0.08,0.09,0.11,0.12,.13,.14,.15,.16,.17,.18,.19]:
        si = probe_sample_interval(True)
        print("SI:",si,"cat:",args.category_id,"dr",dr)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=dr, sample_interval=si,c2s_ratio=args.c2s_ratio,)

        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=dr, sample_interval=0,c2s_ratio=args.c2s_ratio)
        
        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
        with open(f'MCVC-IA-OLFT.dr.log','a') as f:
            f.write(f'{dr},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si}\n')

for cl in range(4):
    for c2s in [0.7,0.87,1.33]:
        si_no_compression = probe_sample_interval(False)
        args.sample_interval = probe_sample_interval(True)
        print("SI:",si_no_compression,args.sample_interval,"cat:",args.category_id,"c2s",c2s)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=args.sample_interval,c2s_ratio=c2s)

        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=c2s)
        
        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)

        # record stat
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.c2s.log','a') as f:
            f.write(f'{c2s},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{args.sample_interval},{si_after_training}\n')

for cl in range(4):
    for mps in [1,5,10,15,20]:
        si_no_compression = probe_sample_interval(False)
        si = probe_sample_interval(True)
        print("SI:",si_no_compression,si,"cat:",args.category_id,"mps",mps)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio,max_pool_size=mps)

        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        
        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)

        # record stat
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.mps.log','a') as f:
            f.write(f'{mps},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{si},{si_after_training}\n')
exit(0)

for cl in range(4):
    si_no_compression = probe_sample_interval(False)
    si = probe_sample_interval(True)
    for sr in [1e-5,1e-4,0.001,0.01,0.1,1]:
        print("Vary SR:","cat:",args.category_id,"si",si,"sr",sr)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views,sample_ratio=sr)

        _, stats0 = train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.sr.log','a') as f:
            f.write(f'{sr},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{stats0[1]:.4f},{stats0[2]:.4f},{si_no_compression},{si},{si_after_training}\n')

for cl in range(4):
    for si in [0,1,10,100,1000]:
        print("Vary SI:","cat:",args.category_id,"si",si,"sr",args.sample_ratio)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)
        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)
        _, stats0 = train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
        with open(f'MCVC-IA-OLFT.si.log','a') as f:
            f.write(f'{si},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{stats0[1]:.4f},{stats0[2]:.4f}\n')

for cl in range(4):

    si_no_compression = probe_sample_interval(False)
    si = probe_sample_interval(True)
    print("SI:",si_no_compression,args.sample_interval,"cat:",args.category_id,
            "c2s",args.c2s_ratio,"si",si,"sr",args.sample_ratio,'NV:',args.num_views)

    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
        data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
    train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
        transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
        data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)

    model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                args.category_id, onlydecoder=args.onlydecoder,
                                                            num_views=train_dataset.num_views)
    model.real_compression = True
    train(0, model, train_dataset, optimizer, test_dataset=test_dataset,codec=args.codec,print_header=[cl,])

for cl in range(4):
    sr_list = [0,0.001,0.01,0.1,1]
    for sr in sr_list:
        si_no_compression = probe_sample_interval(False,sample_ratio=sr)
        si = probe_sample_interval(True,sample_ratio=sr)
        print("Vary SISR (fix 1/100 bw usage):","cl:",cl,"cat:",args.category_id,"si",si,"sr",sr)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                num_views=train_dataset.num_views)

        _, stats0 = train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer,sample_ratio=sr)
        with open(f'MCVC-IA-OLFT.sisr.log','a') as f:
            f.write(f'{sr},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{stats0[1]:.4f},{stats0[2]:.4f},{si_no_compression},{si},{si_after_training}\n')


for cl in range(4):
    for use_attn,load_with_copy in [(False,True),(False,False),(True,True),(True,False)]:

        si_no_compression = probe_sample_interval(False)
        si = probe_sample_interval(True)
        print("SI:",si_no_compression,"cat:",args.category_id,
              "c2s",args.c2s_ratio,"si",si,"sr",args.sample_ratio,args.num_views)

        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=0,c2s_ratio=args.c2s_ratio)
        train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='all',
            transform=shared_transforms,category_id=args.category_id,num_views=args.num_views,
            data_ratio=args.data_ratio, sample_interval=si,c2s_ratio=args.c2s_ratio)

        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,cl, 
                                                                    args.category_id, onlydecoder=args.onlydecoder,
                                                                    num_views=train_dataset.num_views,use_attn=use_attn,
                                                                    load_pretrain=False,load_with_copy=load_with_copy)

        train(0, model, train_dataset, optimizer)
        _, stats = test(0, model, test_dataset)
        
        si_after_training = probe_sample_interval(probe_dataset=train_dataset,probe_model=model,optimizer=optimizer)
        with open(f'MCVC-IA-OLFT.ablation.log','a') as f:
            f.write(f'{use_attn},{load_with_copy},{cl},{stats[0]:.4f},{stats[1]:.4f},{stats[2]:.4f},{si_no_compression},{si},{si_after_training}\n')
