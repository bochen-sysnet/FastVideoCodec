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

from models import get_codec_model,parallel_compression,update_training,compress_whole_video
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

from dataset import VideoDataset, FrameDataset

parser = argparse.ArgumentParser(description='PyTorch EAVC Training')
parser.add_argument('--dataset', type=str, default='UVG', choices=['UVG','MCL-JCV','UVG/2k','MCL-JCV/2k'],
                    help='evaluating dataset (default: UVG)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--codec', type=str, default='Base',
                    help='name of codec')
parser.add_argument('--device', default=0, type=int,
                    help="GPU ID")
parser.add_argument('--epoch', type=int, nargs='+', default=[0,10],
                    help='Begin and end epoch')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--alpha', type=float, default=1.,
                    help='AD rate')
parser.add_argument('--resolution', type=int, default=256, choices=[256,720,1080,2160],
                    help='Frame resolution') # or 960,1920; 720,1280;1920,3840

args = parser.parse_args()

# OPTION
CODEC_NAME = args.codec #'Base-ER2'
SAVE_DIR = f'backup/{CODEC_NAME}'
loss_type = 'P'
compression_level = 0 # 0,1,2,3
RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth'
LEARNING_RATE = args.lr
WEIGHT_DECAY = 5e-4
BEGIN_EPOCH = args.epoch[0]
END_EPOCH = args.epoch[1]
WARMUP_EPOCH = 5
device = args.device
STEPS = []

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

####### Create model
seed = int(time.time())
#seed = int(0)
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

####### Create optimizer
# ---------------------------------------------------------------
parameters = [p for n, p in model.named_parameters()]
optimizer = torch.optim.Adam([{'params': parameters}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# initialize best score
best_codec_score = [1,0,0]

####### Load yowo model
# ---------------------------------------------------------------
# try to load codec model 
if CODEC_NAME in ['x265', 'x264', 'RAW']:
    # nothing to load
    print("No need to load for ", CODEC_NAME)
elif CODEC_NAME in ['SSF-Official']:
    print('Official model loaded.')
elif CODEC_NAME in ['DVC-pretrained']:
    pretrained_model_path = 'DVC/snapshot/2048.model'
    from DVC.net import load_model
    load_model(model, pretrained_model_path)
elif CODEC_NAME in ['ELFVC']:
    # pretrained_model_path = '/home/monet/research/FastVideoCodec/backup/SSF-Official/SSF-Official-0P_best.pth'
    checkpoint = torch.load(RESUME_CODEC_PATH,map_location=torch.device('cuda:'+str(device)))
    # BEGIN_EPOCH = checkpoint['epoch'] + 1
    best_codec_score = checkpoint['score']
    load_state_dict_whatever(model, checkpoint['state_dict'])
    # load_state_dict_all(model, checkpoint['state_dict'])
    print("Loaded model ",CODEC_NAME, ':', best_codec_score)
elif RESUME_CODEC_PATH and os.path.isfile(RESUME_CODEC_PATH):
    print("Loading all for ", CODEC_NAME, 'from',RESUME_CODEC_PATH)
    checkpoint = torch.load(RESUME_CODEC_PATH,map_location=torch.device('cuda:'+str(device)))
    # BEGIN_EPOCH = checkpoint['epoch'] + 1
    best_codec_score = checkpoint['score']
    load_state_dict_all(model, checkpoint['state_dict'])
    print("Loaded model codec score: ", checkpoint['score'])
    del checkpoint
elif 'Base' in CODEC_NAME:
    # load what exists
    pretrained_model_path = f'DVC/snapshot/512.model'#f'backup/LSVC-A/LSVC-A-{compression_level}P_best.pth'
    checkpoint = torch.load(pretrained_model_path,map_location=torch.device('cuda:'+str(device)))
    if 'state_dict' in checkpoint.keys():
        load_state_dict_whatever(model, checkpoint['state_dict'])
        best_codec_score = checkpoint['score']
    else:
        # model.load_state_dict(checkpoint)
        load_state_dict_whatever(model, checkpoint)
    del checkpoint
    print("Load whatever exists for",RESUME_CODEC_PATH,'from',RESUME_CODEC_PATH,best_codec_score)
    # with open(f'DVC/snapshot/512.model', 'rb') as f:
    #    pretrained_dict = torch.load(f)
    #    load_state_dict_only(model, pretrained_dict, 'warpnet')
    #    load_state_dict_only(model, pretrained_dict, 'opticFlow')
       # del pretrained_dict
else:
    print("Cannot load model codec", RESUME_CODEC_PATH)
print("===================================================================")

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
    img_loss_module = AverageMeter()
    be_loss_module = AverageMeter()
    be_res_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    I_module = AverageMeter()
    all_loss_module = AverageMeter()
    aux_loss_module = AverageMeter()
    aux2_loss_module = AverageMeter()
    aux3_loss_module = AverageMeter()
    aux4_loss_module = AverageMeter()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    batch_size = 7
    ds_size = len(train_dataset)
    
    model.train()
    if model.name == 'DVC-pretrained':model.eval()
    update_training(model,epoch,warmup_epoch=WARMUP_EPOCH)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, 
                                               num_workers=8, drop_last=True, pin_memory=True)
    
    train_iter = tqdm(train_loader)
    for batch_idx,data in enumerate(train_iter):
        data = data[0].cuda(device)
        l = data.size(0)-1
        
        # run model
        _,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux_loss2,aux_loss3,aux_loss4 = parallel_compression(args,model,data,True)

        # record loss
        all_loss_module.update(loss.cpu().data.item(), l)
        img_loss_module.update(img_loss, l)
        be_loss_module.update(be_loss, l)
        be_res_loss_module.update(be_res_loss, l)
        if not math.isinf(psnr):
            psnr_module.update(psnr,l)
            I_module.update(I_psnr)
        aux_loss_module.update(aux_loss, l)
        aux2_loss_module.update(aux_loss2, l)
        aux3_loss_module.update(aux_loss3, l)
        aux4_loss_module.update(aux_loss4, l)
        
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
            f"L:{all_loss_module.val:.2f} ({all_loss_module.avg:.2f}). "
            f"I:{img_loss_module.val:.2f} ({img_loss_module.avg:.2f}). "
            f"B:{be_loss_module.val:.2f} ({be_loss_module.avg:.2f}). "
            f"P:{psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            # f"R:{be_res_loss_module.val:.2f} ({be_res_loss_module.avg:.2f}). "
            # f"I:{I_module.val:.2f} ({I_module.avg:.2f}). "
            f"1:{aux_loss_module.val:.4f} ({aux_loss_module.avg:.4f}). "
            f"2:{aux2_loss_module.val:.4f} ({aux2_loss_module.avg:.4f}). "
            f"3:{aux3_loss_module.val:.4f} ({aux3_loss_module.avg:.4f}). "
            f"4:{aux4_loss_module.val:.4f} ({aux4_loss_module.avg:.4f}). ")

        # clear result every 1000 batches
        if batch_idx % 1000 == 0 and batch_idx>0: # From time to time, reset averagemeters to see improvements
            img_loss_module.reset()
            aux_loss_module.reset()
            be_loss_module.reset()
            be_res_loss_module.reset()
            all_loss_module.reset()
            psnr_module.reset()
            aux2_loss_module.reset() 
            I_module.reset()    
            
        if batch_idx % 10000 == 0 and batch_idx>0:
            if True:
                print('Testing at batch_idx %d' % (batch_idx))
                score = test(epoch, model, test_dataset)
                
                is_best = score[0] <= best_codec_score[0] and score[1] >= best_codec_score[1]
                if is_best:
                    print("New best score: ", score, ". Previous: ", best_codec_score)
                    best_codec_score = score
                else:
                    print(score)
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score}
                save_checkpoint(state, is_best, SAVE_DIR, CODEC_NAME, loss_type, compression_level)
                model.train()
            else:
                print('')
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': best_codec_score}
                save_checkpoint(state, False, SAVE_DIR, CODEC_NAME, loss_type, compression_level)
    return best_codec_score
    
def test(epoch, model, test_dataset):
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
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
            data = torch.stack(data, dim=0).cuda(device)
            l = data.size(0)
            
            # compress GoP
            if l>fP+1:
                com_imgs,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux_loss2,_,_ = parallel_compression(args,model,torch.flip(data[:fP+1],[0]),True)
                ba_loss_module.update(be_loss, fP+1)
                psnr_module.update(psnr,fP+1)
                data[fP:fP+1] = com_imgs[0:1]
                com_imgs,loss,img_loss,be_loss,be_res_loss,psnr,_,aux_loss,aux_loss2,_,_ = parallel_compression(args,model,data[fP:],False)
                ba_loss_module.update(be_loss, l-fP-1)
                psnr_module.update(psnr,l-fP-1)
            else:
                com_imgs,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux_loss2,_,_ = parallel_compression(args,model,torch.flip(data,[0]),True)
                ba_loss_module.update(be_loss, l)
                psnr_module.update(psnr,l)
                
        # show result
        test_iter.set_description(
            f"{data_idx:6}. "
            f"BA: {ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
            f"P: {psnr_module.val:.4f} ({psnr_module.avg:.4f}). "
            f"I: {I_psnr:.4f}")
            
        # clear input
        data = []
        
    test_dataset.reset()
    return [ba_loss_module.avg,psnr_module.avg]
                        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    LEARNING_RATE = 1e-4
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
          
train_dataset = FrameDataset('../dataset/vimeo', frame_size=256) 
test_dataset = VideoDataset(f'../dataset/{args.dataset}', args.resolution)
# test_dataset2 = VideoDataset('../dataset/MCL-JCV', frame_size=(256,256))
if args.evaluate:
    score = test(0, model, test_dataset)
    exit(0)

for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
    # Adjust learning rate
    r = adjust_learning_rate(optimizer, epoch)
    
    print('training at epoch %d, r=%.2f' % (epoch,r))
    best_codec_score = train(epoch, model, train_dataset, optimizer, best_codec_score, test_dataset)
    
    print('testing at epoch %d' % (epoch))
    score = test(epoch, model, test_dataset)
    
    is_best = score[0] <= best_codec_score[0] and score[1] >= best_codec_score[1]
    if is_best:
        print("New best score is achieved: ", score, ". Previous score was: ", best_codec_score)
        best_codec_score = score
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'score': score}
    save_checkpoint(state, is_best, SAVE_DIR, CODEC_NAME, loss_type, compression_level)
    print('Weights are saved to backup directory: %s' % (SAVE_DIR), 'score:',score)
    # test(epoch, model, test_dataset2)