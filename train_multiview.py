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

from dataset import SynVideoDataset, FrameDataset, MultiViewVideoDataset
import pytorch_msssim

parser = argparse.ArgumentParser(description='PyTorch EAVC Training')
parser.add_argument('--batch-size', default=2, type=int,
                    help="batch size")
parser.add_argument('--benchmark', action='store_true',
                    help='benchmark model on validation set')
parser.add_argument('--super-batch', default=16, type=int,
                    help="super batch size")
parser.add_argument('--num_views', default=0, type=int,
                    help="number of views")
parser.add_argument('--debug', action='store_true',
                    help='debug model on validation set')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--codec', type=str, default='MCVC-IA0',
                    help='name of codec')
parser.add_argument('--category', default=0, type=int,
                    help="Category ID")
parser.add_argument('--device', default=0, type=int,
                    help="GPU ID")
parser.add_argument('--epoch', type=int, nargs='+', default=[0,1000],
                    help='Begin and end epoch')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--loss-type', type=str, default='P',
                    help='PSNR or MS-SSIM')
parser.add_argument('--width', type=int, default=256,
                    help='Frame width') 
parser.add_argument('--height', type=int, default=256,
                    help='Frame height') 
parser.add_argument('--compression_level', default=0, type=int,
                    help="Compression level")
parser.add_argument('--max_files', default=0, type=int,
                    help="Maximum loaded files")
parser.add_argument('--resume', type=str, default='',
                    help='Resume path')
parser.add_argument('--alpha', type=float, default=100,
                    help='Controlling norm scale')
parser.add_argument('--resilience', default=10, type=int,
                    help="Number of losing views to tolerate")
parser.add_argument('--force-resilience', default=0, type=int,
                    help="Force the number of losing views in training/evaluation")
parser.add_argument("--fP", type=int, default=15, help="The number of forward P frames")
parser.add_argument("--bP", type=int, default=0, help="The number of backward P frames")
parser.add_argument('--level-range', type=int, nargs='+', default=[0,4])
parser.add_argument('--frame-comb', default=0, type=int, help="Frame combination method. 0: naive. 1: spatial. 2: temporal.")

args = parser.parse_args()

# OPTION
CODEC_NAME = args.codec
SAVE_DIR = f'backup/{CODEC_NAME}'
loss_type = args.loss_type
LEARNING_RATE = args.lr
WEIGHT_DECAY = 5e-4
device = args.device
STEPS = []

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

seed = int(time.time())
# seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

def get_model_n_optimizer_n_score_from_level(codec_name,compression_level,category_id):
    # codec model
    model = get_codec_model(codec_name, 
                            loss_type=loss_type, 
                            compression_level=compression_level,
                            use_split=False,
                            num_views=test_dataset.num_views,
                            resilience=args.resilience)
    model.force_resilience = args.force_resilience
    model = model.cuda(device)

    # initialize best score
    best_codec_score = 100

    def load_from_path(path):
        print("Loading for ", codec_name, 'from',path)
        checkpoint = torch.load(path,map_location=torch.device('cuda:'+str(device)))
        # load_state_dict_all(model, checkpoint['state_dict'])
        load_state_dict_whatever(model, checkpoint['state_dict'])
        if isinstance(checkpoint['score'],float):
            best_codec_score = checkpoint['score']
            print("Loaded model codec score: ", checkpoint['score'])
        if 'stats' in checkpoint:
            print("Loaded model codec stat: ", checkpoint['stats'])
        del checkpoint

    paths = []
    paths += f'{SAVE_DIR}/{codec_name}-{compression_level}{loss_type}_vid{category_id}_best.pth'
    paths += f'{SAVE_DIR}/{codec_name}-{compression_level}{loss_type}_vid{category_id}_ckpt.pth'
    if codec_name == 'MCVC-IA':
        paths += f'{SAVE_DIR}/MCVC-IA0-{compression_level}{loss_type}_vid{category_id}_best.pth'
    if loss_type == 'M':
        paths += f'{SAVE_DIR}/{codec_name}-{compression_level}P_vid{category_id}_best.pth'
    paths += f'{SAVE_DIR}/{codec_name}-0{loss_type}_vid{category_id}_ckpt.pth'
    for pth in paths:
        if os.path.isfile(pth):
            load_from_path(pth)
            break

    # create optimizer
    parameters = [p for n, p in model.named_parameters()]
    parameter_names = [n for n, p in model.named_parameters()]
    optimizer = torch.optim.Adam([{'params': parameters}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    pytorch_total_params = sum(p.numel() for p in parameters)
    print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    return model, optimizer, best_codec_score


def metrics_per_gop(out_dec, raw_frames):
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
        x = raw_frames[frame_idx]
        for likelihood_name in ['keyframe', 'motion', 'residual']:
            if likelihood_name in likelihoods:
                var_like = likelihoods[likelihood_name]
                bits = torch.sum(torch.clamp(-1.0 * torch.log(var_like["y"] + 1e-5) / math.log(2.0), 0, 50)) + \
                        torch.sum(torch.clamp(-1.0 * torch.log(var_like["z"] + 1e-5) / math.log(2.0), 0, 50))
        if args.loss_type == 'P':
            if non_zero_indices is None:
                mseloss = torch.mean((x_hat - x).pow(2))
            else:
                mseloss = torch.mean((x_hat[non_zero_indices] - x[non_zero_indices]).pow(2))
        else:
            if non_zero_indices is None:
                mseloss = 1 - pytorch_msssim.ms_ssim(x_hat, x)
            else:
                mseloss = 1 - pytorch_msssim.ms_ssim(x_hat[non_zero_indices], x[non_zero_indices])
        psnr = 10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(raw_frames.device)

        pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = bits / pixels
        total_bpp += bpp
        total_psnr += psnr
        total_mse += mseloss
        frame_idx += 1

    return total_mse/frame_idx,total_bpp/frame_idx,total_psnr/frame_idx,completeness
        
def train(epoch, model, train_dataset, best_codec_score, optimizer):
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    all_loss_module = AverageMeter()
    psnr_vs_resilience = [AverageMeter() for _ in range(train_dataset.num_views)]
    bpp_vs_resilience = [AverageMeter() for _ in range(train_dataset.num_views)]
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ds_size = len(train_dataset)
    
    model.train()
    # multi-view dataset must be single batch in loader 
    # single view dataset set batch size to view numbers in loader in test
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                               num_workers=8, drop_last=True, pin_memory=True)
    
    train_iter = tqdm(train_loader)
    for batch_idx,data in enumerate(train_iter):
        b,g,v,c,h,w = data.size()
        data = data.permute(1,0,2,3,4,5).reshape(g,b*v,c,h,w).cuda(device)
        
        # run model
        out_dec = model(data)
        mse, bpp, psnr, completeness = metrics_per_gop(out_dec, data)
        loss = model.r*mse + bpp
        
        ba_loss_module.update(bpp.cpu().data.item())
        psnr_module.update(psnr.cpu().data.item())
        img_loss_module.update(mse.cpu().data.item())
        all_loss_module.update(loss.cpu().data.item())
        
        # backward
        scaler.scale(loss).backward()
        if batch_idx % args.super_batch == args.super_batch-1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # add metrics
        resi = int(np.round(train_dataset.num_views * (1 - completeness)))
        psnr_vs_resilience[resi].update(psnr.cpu().data.item())
        bpp_vs_resilience[resi].update(bpp.cpu().data.item())
                
        # metrics string
        metrics_str = ""
        for i,(psnr,bpp) in enumerate(zip(psnr_vs_resilience,bpp_vs_resilience)):
            metrics_str += f"{psnr.count}:{psnr.avg:.2f},{bpp.avg:.3f}. "

        # show result
        train_iter.set_description(
            f"{epoch} {batch_idx:6}. "
            f"L:{all_loss_module.val:.4f} ({all_loss_module.avg:.4f}). "
            f"I:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.2f} ({psnr_module.avg:.2f}). " + metrics_str)

    return best_codec_score
    
def test(epoch, model, test_dataset, print_header=None):
    model.eval()
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    psnr_vs_resilience = [AverageMeter() for _ in range(test_dataset.num_views)]
    bpp_vs_resilience = [AverageMeter() for _ in range(test_dataset.num_views)]
    ds_size = len(test_dataset)
    
    data = []
    # test must use a batch size equal to the number of views/cameras
    test_iter = tqdm(range(ds_size))
    eof = False
    for data_idx,_ in enumerate(test_iter):
        data = test_dataset[data_idx].cuda(device)
            
        with torch.no_grad():
            out_dec = model(data)
            mse, bpp, psnr, completeness = metrics_per_gop(out_dec, data)
            
            ba_loss_module.update(bpp.cpu().data.item())
            psnr_module.update(psnr.cpu().data.item())
            img_loss_module.update(mse.cpu().data.item())

        # add metrics
        resi = int(np.round(train_dataset.num_views * (1 - completeness)))
        psnr_vs_resilience[resi].update(psnr.cpu().data.item())
        bpp_vs_resilience[resi].update(bpp.cpu().data.item())
                
        # metrics string
        metrics_str = ""
        for i,(psnr,bpp) in enumerate(zip(psnr_vs_resilience,bpp_vs_resilience)):
            metrics_str += f"{psnr.count}:{psnr.avg:.2f},{bpp.avg:.3f}. "

        # show result
        test_iter.set_description(
            f"{epoch} {data_idx:6}. "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.4f} ({psnr_module.avg:.4f}). "
            f"IL:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). " + metrics_str)

        if print_header is not None:
            with open(f'{args.codec}-{args.frame_comb}.log','a') as f:
                f.write(f'{print_header[0]},{print_header[1]},{ba_loss_module.val:.4f},{psnr_module.val:.4f}\n')
        if args.debug and data_idx == 9:exit(0)
    # test_dataset.reset()        
    return ba_loss_module.avg+model.r*img_loss_module.avg, [ba_loss_module.avg,psnr_module.avg]

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
    print('Saved to:',ckpt_filename)
    if is_best:
        shutil.copyfile(ckpt_filename, best_filename)

    with open(f'{directory}/log.txt','a+') as f:
        f.write(f'{category_id},{compression_level},{epoch},{bpp},{psnr},{score}\n')

def static_simulation_x26x_multicam(args,test_dataset,category_id):
    ds_size = len(test_dataset)
    quality_levels = [7,11,15,19,23,27,31,35]# 15,19,23,27
    
    Q_list = quality_levels[args.level_range[0]:args.level_range[1]]
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
            # with open(f'{args.codec}-{args.frame_comb}.log','a') as f:
            #     f.write(f'{category_id},{lvl},{ba_loss_module.val:.4f},{psnr_module.val:.4f},{msssim_module.val:.4f}\n')

def static_simulation_model_multicam(args, test_dataset,category_id):
    for lvl in range(args.level_range[0],args.level_range[1]):
        model, _, _ = get_model_n_optimizer_n_score_from_level(CODEC_NAME,lvl,args.category_id)
        model.eval()
        test(0, model, test_dataset, print_header=[category_id,lvl])

if args.benchmark:
    for category_id in range(5):
        shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
        test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',transform=shared_transforms,\
            category_id=category_id,num_views=args.num_views)
        if 'x26' in args.codec:
            static_simulation_x26x_multicam(args, test_dataset, category_id)
        else:
            static_simulation_model_multicam(args, test_dataset, category_id)
        break

if args.evaluate:
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',transform=shared_transforms,category_id=args.category,num_views=args.num_views)

    model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,args.compression_level,args.category_id)
    score, stats = test(0, model, test_dataset)
    exit(0)

for category_id in range(5):
    shared_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
    train_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='train',transform=shared_transforms,category_id=category_id,num_views=args.num_views)
    test_dataset = MultiViewVideoDataset('../dataset/multicamera/',split='test',transform=shared_transforms,category_id=category_id,num_views=args.num_views)

    start = 0# if category_id>0 else 1
    for compression_level in range(start,4):
        model, optimizer, best_codec_score = get_model_n_optimizer_n_score_from_level(CODEC_NAME,compression_level, category_id)

        cvg_cnt = 0
        BEGIN_EPOCH = args.epoch[0]
        END_EPOCH = args.epoch[1]
        for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
            best_codec_score = train(epoch, model, train_dataset, best_codec_score, optimizer)
            
            score, stats = test(epoch, model, test_dataset)
            
            is_best = score <= best_codec_score
            if is_best:
                print("New best", stats, "Score:", score, ". Previous: ", best_codec_score)
                best_codec_score = score
                cvg_cnt = 0
            else:
                cvg_cnt += 1
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score, 'stats': stats}
            save_checkpoint(state, is_best, SAVE_DIR, CODEC_NAME, loss_type, compression_level, category_id)
            if cvg_cnt == 10:break