from __future__ import print_function
import os
import sys
import time
import math
import random

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

from models import get_codec_model,parallel_compression,update_training,compress_whole_video
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

from dataset import VideoDataset, FrameDataset

# OPTION
BACKUP_DIR = 'backup'
CODEC_NAME = 'SPVC'
loss_type = 'P'
compression_level = 2 # 0,1,2,3
RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth'
#RESUME_CODEC_PATH = '../YOWO/backup/ucf24/yowo_ucf24_16f_SPVC_ckpt.pth'
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
BEGIN_EPOCH = 1
END_EPOCH = 10
WARMUP_EPOCH = 1
USE_VIMEO = True

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
        
def train(epoch, model, train_dataset, optimizer, best_codec_score, test_dataset):
    aux_loss_module = AverageMeter()
    img_loss_module = AverageMeter()
    be_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    msssim_module = AverageMeter()
    all_loss_module = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch_size = 7
    ds_size = len(train_dataset)
    
    model.train()
    update_training(model,epoch)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                               num_workers=8, drop_last=True, pin_memory=True)
    
    train_iter = tqdm(train_loader)
    for batch_idx,data in enumerate(train_iter):
        if batch_idx<=45000:continue
        data = data[0].cuda()
        l = data.size(0)-1
        
        # run model
        _,img_loss_list,bpp_est_list,aux_loss_list,psnr_list,msssim_list,_ = parallel_compression(model,data,True)
        
        # aggregate loss
        be_loss = torch.stack(bpp_est_list,dim=0).mean(dim=0)
        aux_loss = torch.stack(aux_loss_list,dim=0).mean(dim=0)
        img_loss = torch.stack(img_loss_list,dim=0).mean(dim=0)
        psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
        msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
        loss = model.loss(img_loss,be_loss,aux_loss)
        
        # record loss
        aux_loss_module.update(aux_loss.cpu().data.item(), l)
        img_loss_module.update(img_loss.cpu().data.item(), l)
        be_loss_module.update(be_loss.cpu().data.item(), l)
        psnr_module.update(psnr.cpu().data.item(),l)
        msssim_module.update(msssim.cpu().data.item(), l)
        all_loss_module.update(loss.cpu().data.item(), l)
        
        # backward
        scaler.scale(loss).backward()
        # update model after compress each video
        if batch_idx%10 == 0 and batch_idx > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # show result
        train_iter.set_description(
            f"{batch_idx:6}. "
            f"IL: {img_loss_module.val:.2f} ({img_loss_module.avg:.2f}). "
            f"BE: {be_loss_module.val:.2f} ({be_loss_module.avg:.2f}). "
            f"AX: {aux_loss_module.val:.2f} ({aux_loss_module.avg:.2f}). "
            f"AL: {all_loss_module.val:.2f} ({all_loss_module.avg:.2f}). "
            f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). "
            f"I: {float(psnr_list[0]):.2f}")

        # clear result every 1000 batches
        if batch_idx % 1000 == 0 and batch_idx>0: # From time to time, reset averagemeters to see improvements
            img_loss_module.reset()
            aux_loss_module.reset()
            be_loss_module.reset()
            all_loss_module.reset()
            psnr_module.reset()
            msssim_module.reset()   
            
        if batch_idx % 5000 == 0 and batch_idx>0:
            print('testing at batch_idx %d' % (batch_idx))
            score = test(epoch, model, test_dataset)
            
            is_best = isinstance(best_codec_score,list) and (score[0] <= best_codec_score[0]) and (score[1] >= best_codec_score[1])
            if is_best:
                print("New best score is achieved: ", score, ". Previous score was: ", best_codec_score)
                best_codec_score = score
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score}
            save_checkpoint(state, is_best, BACKUP_DIR, CODEC_NAME, loss_type, compression_level)
            model.train()
    
def test(epoch, model, test_dataset):
    aux_loss_module = AverageMeter()
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    be_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    msssim_module = AverageMeter()
    all_loss_module = AverageMeter()
    ds_size = len(test_dataset)
    
    model.eval()
    
    fP,bP = 6,6
    GoP = fP+bP+1
    
    data = []
    test_iter = tqdm(range(ds_size))
    for data_idx,_ in enumerate(test_iter):
        frame,eof = test_dataset[data_idx]
        data.append(transforms.ToTensor()(frame))
        if len(data) < GoP and not eof:
            continue
            
        with torch.no_grad():
            data = torch.stack(data, dim=0).cuda()
            l = data.size(0)
            
            # compress GoP
            if l>fP+1:
                com_imgs,img_loss_list1,bpp_est_list1,aux_loss_list1,psnr_list1,msssim_list1,bpp_act_list1 = parallel_compression(model,torch.flip(data[:fP+1],[0]),True)
                data[fP:fP+1] = com_imgs[0:1]
                _,img_loss_list2,bpp_est_list2,aux_loss_list2,psnr_list2,msssim_list2,bpp_act_list2 = parallel_compression(model,data[fP:],False)
                img_loss_list = img_loss_list1[::-1] + img_loss_list2
                aux_loss_list = aux_loss_list1[::-1] + aux_loss_list2
                psnr_list = psnr_list1[::-1] + psnr_list2
                msssim_list = msssim_list1[::-1] + msssim_list2
                bpp_act_list = bpp_act_list1[::-1] + bpp_act_list2
                bpp_est_list = bpp_est_list1[::-1] + bpp_est_list2
            else:
                _,img_loss_list,bpp_est_list,aux_loss_list,psnr_list,msssim_list,bpp_act_list = parallel_compression(model,torch.flip(data,[0]),True)
                
            # aggregate loss
            ba_loss = torch.stack(bpp_act_list,dim=0).mean(dim=0)
            be_loss = torch.stack(bpp_est_list,dim=0).mean(dim=0)
            aux_loss = torch.stack(aux_loss_list,dim=0).mean(dim=0)
            img_loss = torch.stack(img_loss_list,dim=0).mean(dim=0)
            psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
            msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
            loss = model.loss(img_loss,ba_loss,aux_loss)
            
            # record loss
            aux_loss_module.update(aux_loss.cpu().data.item(), l)
            img_loss_module.update(img_loss.cpu().data.item(), l)
            ba_loss_module.update(ba_loss.cpu().data.item(), l)
            be_loss_module.update(ba_loss.cpu().data.item(), l)
            psnr_module.update(psnr.cpu().data.item(),l)
            msssim_module.update(msssim.cpu().data.item(), l)
            all_loss_module.update(loss.cpu().data.item(), l)
        
        # show result
        test_iter.set_description(
            f"{data_idx:6}. "
            f"IL: {img_loss_module.val:.2f} ({img_loss_module.avg:.2f}). "
            f"BA: {ba_loss_module.val:.2f} ({ba_loss_module.avg:.2f}). "
            f"BE: {be_loss_module.val:.2f} ({be_loss_module.avg:.2f}). "
            f"AX: {aux_loss_module.val:.2f} ({aux_loss_module.avg:.2f}). "
            f"AL: {all_loss_module.val:.2f} ({all_loss_module.avg:.2f}). "
            f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). "
            f"I: {float(max(psnr_list)):.2f}")
            
        # clear input
        data = []
        
    test_dataset.reset()
    return [ba_loss_module.avg,psnr_module.avg,msssim_module.avg]
                        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    LEARNING_RATE = 1e-4
    LR_DECAY_RATE = 0.1
    STEPS = []
    steps = [s for s in STEPS if s<0] if epoch<0 else [s for s in STEPS if s>=0]
    r = (LR_DECAY_RATE ** (sum(epoch >= np.array(steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= r
    return r
    
def train_ucf(epoch, model_codec, train_dataset, optimizer, best_codec_score):
    t0 = time.time()
    aux_loss_module = AverageMeter()
    img_loss_module = AverageMeter()
    be_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    msssim_module = AverageMeter()
    all_loss_module = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch_size = 4
    l_loader = len(train_dataset)//batch_size

    model_codec.train()
    # get instructions on training
    update_training(model_codec,epoch,warmup_epoch=WARMUP_EPOCH)
    train_iter = tqdm(range(0,l_loader*batch_size,batch_size))
    frame_idx = []; data = []; target = []; img_loss_list = []; aux_loss_list = []
    bpp_est_list = []; psnr_list = []; msssim_list = []
    for batch_idx,_ in enumerate(train_iter):
        # align batches
        for j in range(batch_size):
            data_idx = batch_idx*batch_size+j
            # compress one batch of the data
            train_dataset.preprocess(data_idx, model_codec)
            # read one clip
            f,d,t,additional = train_dataset[data_idx]
            frame_idx.append(f-1)
            data.append(d)
            target.append(t)
            bpp_est_list.append(additional['bpp_est'])
            aux_loss_list.append(additional['aux_loss'])
            img_loss_list.append(additional['img_loss'])
            psnr_list.append(additional['psnr'])
            msssim_list.append(additional['msssim'])
            if train_dataset.last_frame or additional['end_of_batch']:
                # we split if the batch of compression ends or if the video ends or if its a i frame
                # if is the end of a video
                data = torch.stack(data, dim=0).cuda()
                target = torch.stack(target, dim=0)
                l = len(frame_idx)
                with autocast():
                    be_loss = torch.stack(bpp_est_list,dim=0).mean(dim=0)
                    aux_loss = torch.stack(aux_loss_list,dim=0).mean(dim=0)
                    img_loss = torch.stack(img_loss_list,dim=0).mean(dim=0)
                    loss = model_codec.loss(img_loss,be_loss,aux_loss)
                    psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
                    msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
                    aux_loss_module.update(aux_loss.cpu().data.item(), l)
                    img_loss_module.update(img_loss.cpu().data.item(), l)
                    be_loss_module.update(be_loss.cpu().data.item(), l)
                    all_loss_module.update(loss.cpu().data.item(), l)
                    psnr_module.update(psnr.cpu().data.item(),l)
                    msssim_module.update(msssim.cpu().data.item(), l)
                # backward prop
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                # update model after compress each video
                if train_dataset.last_frame:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                # init batch
                frame_idx = []; data = []; target = []; img_loss_list = []; aux_loss_list = []
                bpp_est_list = []; psnr_list = []; msssim_list = []

        # show result
        train_iter.set_description(
            f"Batch: {batch_idx:6}. "
            f"IL: {img_loss_module.val:.2f} ({img_loss_module.avg:.2f}). "
            f"BE: {be_loss_module.val:.2f} ({be_loss_module.avg:.2f}). "
            f"AX: {aux_loss_module.val:.2f} ({aux_loss_module.avg:.2f}). "
            f"AL: {all_loss_module.val:.2f} ({all_loss_module.avg:.2f}). "
            f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). ")
            
        # save result every 1000 batches
        if batch_idx % 10000 == 0: # From time to time, reset averagemeters to see improvements
            print('')
            img_loss_module.reset()
            aux_loss_module.reset()
            be_loss_module.reset()
            all_loss_module.reset()
            psnr_module.reset()
            msssim_module.reset()

    t1 = time.time()
    print('trained with %f samples/s' % (len(train_dataset)/(t1-t0)))

def save_checkpoint(state, is_best, directory, CODEC_NAME, loss_type, compression_level):
    import shutil
    torch.save(state, f'{directory}/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth')
    if is_best:
        shutil.copyfile(f'{directory}/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth',
                        f'{directory}/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth')
                        
#benchmarking()

####### Check backup directory, create if necessary
# ---------------------------------------------------------------
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

####### Create model
#seed = int(time.time())
seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

# codec model .
model = get_codec_model(CODEC_NAME, loss_type=loss_type, compression_level=compression_level)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

####### Create optimizer
# ---------------------------------------------------------------
parameters = [p for n, p in model.named_parameters() if (not n.endswith(".quantiles"))]
aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
optimizer = torch.optim.Adam([{'params': parameters},{'params': aux_parameters, 'lr': 10*LEARNING_RATE}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# initialize best score
best_codec_score = [1,0,0]

####### Load yowo model
# ---------------------------------------------------------------
# try to load codec model 
if CODEC_NAME in ['x265', 'x264', 'RAW']:
    # nothing to load
    print("No need to load for ", CODEC_NAME)
elif CODEC_NAME in ['SCVC']:
    # load what exists
    print("Load whatever exists for",CODEC_NAME)
    pretrained_model_path = "backup/RLVC/RLVC-2P_best.pth"
    checkpoint = torch.load(pretrained_model_path)
    load_state_dict_whatever(model, checkpoint['state_dict'])
    del checkpoint
elif RESUME_CODEC_PATH and os.path.isfile(RESUME_CODEC_PATH):
    print("Loading for ", CODEC_NAME, 'from',RESUME_CODEC_PATH)
    checkpoint = torch.load(RESUME_CODEC_PATH)
    BEGIN_EPOCH = 1#checkpoint['epoch'] + 1
    best_codec_score = checkpoint['score']
    load_state_dict_all(model, checkpoint['state_dict'])
    print("Loaded model codec score: ", checkpoint['score'])
    del checkpoint
else:
    print("Cannot load model codec", RESUME_CODEC_PATH)
print("===================================================================")
    
####### Load dataset
import sys
sys.path.append('../YOWO')
from datasets import list_dataset
BASE_PTH = "../dataset/ucf24"
TRAIN_FILE = "../dataset/ucf24/trainlist.txt"
TRAIN_CROP_SIZE = 224
NUM_FRAMES = 16
SAMPLING_RATE = 1
if USE_VIMEO:
    train_dataset = FrameDataset('../dataset/vimeo') # this dataset might not work?
else:
    train_dataset = list_dataset.UCF_JHMDB_Dataset_codec(BASE_PTH, TRAIN_FILE, dataset='ucf24',
                           shape=(TRAIN_CROP_SIZE, TRAIN_CROP_SIZE),
                           transform=transforms.Compose([transforms.ToTensor()]), 
                           train=True, clip_duration=NUM_FRAMES, sampling_rate=SAMPLING_RATE)
test_dataset = VideoDataset('../dataset/UVG', frame_size=(256,256))
#test_dataset2 = VideoDataset('../dataset/MCL-JCV', frame_size=(256,256))

for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
    # Adjust learning rate
    r = adjust_learning_rate(optimizer, epoch)
    
    print('training at epoch %d, r=%.2f' % (epoch,r))
    if USE_VIMEO:
        train(epoch, model, train_dataset, optimizer, best_codec_score, test_dataset)
    else:
        train_ucf(epoch, model, train_dataset, optimizer, best_codec_score)
    
    print('testing at epoch %d' % (epoch))
    score = test(epoch, model, test_dataset)
    
    is_best = isinstance(best_codec_score,list) and (score[0] <= best_codec_score[0]) and (score[1] >= best_codec_score[1])
    if is_best:
        print("New best score is achieved: ", score, ". Previous score was: ", best_codec_score)
        best_codec_score = score
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score}
    save_checkpoint(state, is_best, BACKUP_DIR, CODEC_NAME, loss_type, compression_level)
    print('Weights are saved to backup directory: %s' % (BACKUP_DIR), 'score:',score)