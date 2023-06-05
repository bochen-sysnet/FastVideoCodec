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

from models import get_codec_model,parallel_compression, AverageMeter
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

from dataset import SynVideoDataset, FrameDataset, MultiViewVideoDataset

parser = argparse.ArgumentParser(description='PyTorch EAVC Training')
parser.add_argument('--dataset', type=str, default='UVG', choices=['UVG','MCL-JCV','UVG/2k','MCL-JCV/2k'],
                    help='evaluating dataset (default: UVG)')
parser.add_argument('--batch_size', default=8, type=int,
                    help="batch size")
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--codec', type=str, default='MCVC',
                    help='name of codec')
parser.add_argument('--category', default=0, type=int,
                    help="Category ID")
parser.add_argument('--device', default=0, type=int,
                    help="GPU ID")
parser.add_argument('--epoch', type=int, nargs='+', default=[0,100],
                    help='Begin and end epoch')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--width', type=int, default=256,
                    help='Frame width') 
parser.add_argument('--height', type=int, default=256,
                    help='Frame height') 
parser.add_argument('--compression_level', default=0, type=int,
                    help="Compression level")
parser.add_argument('--max_files', default=0, type=int,
                    help="Maximum loaded files")
parser.add_argument('--evolve_rounds', default=1, type=int,
                    help="Maximum evolving rounds")
parser.add_argument('--resume', type=str, default='',
                    help='Resume path')
parser.add_argument('--norm', default=2, type=int,
                    help="Norm type")
parser.add_argument('--alpha', type=float, default=100,
                    help='Controlling norm scale')

args = parser.parse_args()

# OPTION
CODEC_NAME = args.codec
SAVE_DIR = f'backup/{CODEC_NAME}'
loss_type = 'P'
compression_level = args.compression_level # 0-7
if args.resume == '':
    RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth'
else:
    RESUME_CODEC_PATH = args.resume
LEARNING_RATE = args.lr
WEIGHT_DECAY = 5e-4
BEGIN_EPOCH = args.epoch[0]
END_EPOCH = args.epoch[1]
device = args.device
STEPS = []

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

####### Create model
seed = int(time.time())
# seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

# codec model .
model = get_codec_model(CODEC_NAME, 
                        loss_type=loss_type, 
                        compression_level=compression_level,
                        use_split=False)
model = model.cuda(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

# initialize best score
best_codec_score = 100

# ---------------------------------------------------------------
# try to load codec model 
if RESUME_CODEC_PATH and os.path.isfile(RESUME_CODEC_PATH):
    print("Loading ckpt for ", CODEC_NAME, 'from',RESUME_CODEC_PATH)
    checkpoint = torch.load(RESUME_CODEC_PATH,map_location=torch.device('cuda:'+str(device)))
    # BEGIN_EPOCH = checkpoint['epoch'] + 1
    if isinstance(checkpoint['score'],float):
        best_codec_score = checkpoint['score']
    # load_state_dict_all(model, checkpoint['state_dict'])
    load_state_dict_whatever(model, checkpoint['state_dict'])
    print("Loaded model codec score: ", checkpoint['score'], checkpoint['stats'])
    if 'stats' in checkpoint:
        print(checkpoint['stats'])
    del checkpoint
print("===================================================================")

def calc_metrics(out_dec,raw_frames):
    frame_idx = 0
    total_bpp = 0
    total_psnr = 0
    total_mse = 0
    pixels = 0
    for x_hat,likelihoods in zip(out_dec['x_hat'],out_dec['likelihoods']):
        x = raw_frames[frame_idx]
        for likelihood_name in ['keyframe', 'motion', 'residual']:
            if likelihood_name in likelihoods:
                var_like = likelihoods[likelihood_name]
                bits = torch.sum(torch.clamp(-1.0 * torch.log(var_like["y"] + 1e-5) / math.log(2.0), 0, 50)) + \
                        torch.sum(torch.clamp(-1.0 * torch.log(var_like["z"] + 1e-5) / math.log(2.0), 0, 50))
        mseloss = torch.mean((x_hat - x).pow(2))
        psnr = 10.0*torch.log(1/mseloss)/torch.log(torch.FloatTensor([10])).squeeze(0).to(raw_frames.device)
        pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = bits / pixels
        total_bpp += bpp
        total_psnr += psnr
        total_mse += mseloss
        frame_idx += 1
    return total_mse/frame_idx,total_bpp/frame_idx,total_psnr/frame_idx
        
def train(epoch, model, train_dataset, best_codec_score, test_dataset):
    # create optimizer
    parameters = [p for n, p in model.named_parameters()]
    lr = LEARNING_RATE
    optimizer = torch.optim.Adam([{'params': parameters}], lr=lr, weight_decay=WEIGHT_DECAY)
    # Adjust learning rate
    adjust_learning_rate(optimizer, epoch)

    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    all_loss_module = AverageMeter()
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
        mse, bpp, psnr = calc_metrics(out_dec, data)
        loss = model.r*mse + bpp
        
        ba_loss_module.update(bpp.cpu().data.item())
        psnr_module.update(psnr.cpu().data.item())
        img_loss_module.update(mse.cpu().data.item())
        all_loss_module.update(loss.cpu().data.item())
        
        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # show result
        train_iter.set_description(
            f"{epoch} {batch_idx:6}. "
            f"L:{all_loss_module.val:.4f} ({all_loss_module.avg:.4f}). "
            f"I:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.2f} ({psnr_module.avg:.2f}). ")

    return best_codec_score
    
def test(epoch, model, test_dataset):
    model.eval()
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    ds_size = len(test_dataset)
    
    data = []
    # test must use a batch size equal to the number of views/cameras
    test_iter = tqdm(range(ds_size))
    eof = False
    for data_idx,_ in enumerate(test_iter):
        data = test_dataset[data_idx].cuda(device)
            
        with torch.no_grad():
            out_dec = model(data)
            mse, bpp, psnr = calc_metrics(out_dec, data)
            
            ba_loss_module.update(bpp.cpu().data.item())
            psnr_module.update(psnr.cpu().data.item())
            img_loss_module.update(mse.cpu().data.item())
                
        # show result
        test_iter.set_description(
            f"{epoch} {data_idx:6}. "
            f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P:{psnr_module.val:.4f} ({psnr_module.avg:.4f}). "
            f"IL:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). ")
    # test_dataset.reset()        
    return ba_loss_module.avg+model.r*img_loss_module.avg, [ba_loss_module.avg,psnr_module.avg]

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    LR_DECAY_RATE = 0.1
    r = (LR_DECAY_RATE ** (sum(epoch >= np.array(STEPS))))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= r
    return r
    
def save_checkpoint(state, is_best, directory, CODEC_NAME, loss_type, compression_level):
    import shutil
    epoch = state['epoch']
    torch.save(state, f'{directory}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth')
    if is_best:
        shutil.copyfile(f'{directory}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth',
                        f'{directory}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth')
          
# define multi-view dataset
# train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=256),transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize(size=(256,256)),transforms.ToTensor()])
train_dataset = MultiViewVideoDataset('../dataset/multicamera/MMPTracking/',split='train',transform=train_transforms,category_id=args.category)
test_dataset = MultiViewVideoDataset('../dataset/multicamera/MMPTracking/',split='test',transform=test_transforms,category_id=args.category)

# view_transforms = [transforms.ToTensor()]

# # Rotation
# angle = 30  # Specify the angle of rotation in degrees

# # Translation
# translate_x = 0.1  # Specify the horizontal translation in pixels
# translate_y = 0.2  # Specify the vertical translation in pixels

# # Define the transformations
# view_transforms += [transforms.Compose([
#     transforms.RandomRotation(angle),
#     transforms.RandomAffine(0, translate=(translate_x, translate_y)),
#     transforms.ToTensor()
# ])]
# train_dataset = FrameDataset('../dataset/vimeo', frame_size=256, view_transforms=view_transforms) 
# test_dataset = SynVideoDataset(f'../dataset/{args.dataset}', (args.height, args.width), args.max_files, view_transforms=view_transforms)


if args.evaluate:
    score, stats = test(0, model, test_dataset)
    exit(0)

for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
    best_codec_score = train(epoch, model, train_dataset, best_codec_score, test_dataset)
    
    score, stats = test(epoch, model, test_dataset)
    
    is_best = score <= best_codec_score
    if is_best:
        print("New best", stats, "Score:", score, ". Previous: ", best_codec_score)
        best_codec_score = score
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score, 'stats': stats}
    save_checkpoint(state, is_best, SAVE_DIR, CODEC_NAME, loss_type, compression_level)
    print('Weights are saved to backup directory: %s' % (SAVE_DIR), 'score:',score)