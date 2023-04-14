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
import threading
import subprocess as sp
import shlex
import struct
import socket
import argparse
from datetime import datetime


from models import get_codec_model,parallel_compression,compress_whole_video
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only
from models import PSNR,MSSSIM

import subprocess

def LoadModel(CODEC_NAME,compression_level = 2,use_split=False):
    loss_type = 'P'
    best_path = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth'
    ckpt_path = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth'

    ####### Codec model 
    model = get_codec_model(CODEC_NAME,loss_type=loss_type,compression_level=compression_level,use_split=use_split,noMeasure=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    if model.name in ['DVC-pretrained','SSF-Official']:
        return model

    ####### Load codec model 
    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint['state_dict'])
        # print("Loaded model codec score: ", checkpoint['score'])
        del checkpoint
    elif os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint['state_dict'])
        # print("Loaded model codec score: ", checkpoint['score'])
        del checkpoint
    elif 'Base' == CODEC_NAME:
        psnr_list = [256,512,1024,2048]
        DVC_ckpt_name = f'DVC/snapshot/{psnr_list[compression_level]}.model'
        checkpoint = torch.load(DVC_ckpt_name,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint)
        # print(f"Loaded model codec from {DVC_ckpt_name}")
        del checkpoint
    else:
        print("Cannot load model codec", CODEC_NAME)
        exit(1)
    return model
    
class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=None):
        self._dataset_dir = os.path.join(root_dir)
        self._frame_size = frame_size
        self._total_frames = 0 # Storing file names in object 
        
        self.get_file_names()
        self._num_files = len(self.__file_names)
        
        self.reset()
        
    def reset(self):
        self._curr_counter = 0
        self._frame_counter = -1 # Count the number of frames used per file
        self._file_counter = -1 # Count the number of files used
        self._dataset_nums = [] # Number of frames to be considered from each file (records+files)
        self._clip = [] # hold video frames
        self._cur_file_names = list(self.__file_names)
        
    @property
    def data(self):
        self._curr_counter+=1
        return self.__getitem__(self._curr_counter)
        
    def __getitem__(self, idx):
        # Get the next dataset if frame number is more than table count
        if not len(self._dataset_nums) or self._frame_counter >= self._dataset_nums[self._file_counter]-1: 
            self.current_file = self._cur_file_names.pop() # get one filename
            cap = cv2.VideoCapture(self.current_file)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            self._clip = []
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                # skip black frames
                if np.sum(img) == 0:continue
                img = Image.fromarray(img)
                if self._frame_size is not None:
                    img = img.resize(self._frame_size) 
                self._clip.append(img)
            self._file_counter +=1
            self._dataset_nums.append(len(self._clip))
            self._frame_counter = 0
        else:
            self._frame_counter+=1
        return self._clip[self._frame_counter],self._frame_counter==self._dataset_nums[self._file_counter]-1
        
    def get_file_names(self):
        print("[log] Looking for files in", self._dataset_dir)  
        self.__file_names = []
        for fn in os.listdir(self._dataset_dir):
            fn = fn.strip("'")
            if fn.split('.')[-1] == 'mp4':
                self.__file_names.append(self._dataset_dir + '/' + fn)
        print("[log] Number of files found {}".format(len(self.__file_names)))  
        
    def __len__(self):
        if not self._total_frames:
            self.count_frames()
        return self._total_frames
        
    def count_frames(self):
        # Count total frames 
        self._total_frames = 0
        for file_name in self.__file_names:
            cap = cv2.VideoCapture(file_name)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                if np.sum(img) == 0:continue
                self._total_frames+=1
            # When everything done, release the video capture object
            cap.release()
        print("[log] Total frames: ", self._total_frames)

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
                        
def static_simulation_x26x(args,test_dataset):
    ds_size = len(test_dataset)
    quality_levels = [7,11,15,19,23,27,31,35]
    
    Q_list = quality_levels[args.level_range[0]:args.level_range[1]]
    for lvl,Q in enumerate(Q_list):
        data = []
        ba_loss_module = AverageMeter()
        psnr_module = AverageMeter()
        msssim_module = AverageMeter()
        test_iter = tqdm(range(ds_size))
        for data_idx,_ in enumerate(test_iter):
            frame,eof = test_dataset[data_idx]
            data.append(frame)
            if not eof:
                continue
            l = len(data)
                
            psnr_list,msssim_list,bpp_act_list,compt,decompt = compress_whole_video(args.task,data,Q,*test_dataset._frame_size, GOP=args.fP + args.bP +1)
            
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
            psnr_list = torch.stack(psnr_list,dim=0).tolist()
            with open(f'{args.task}.log','a') as f:
                f.write(f'{lvl},{ba_loss_module.val:.4f},{compt:.4f},{decompt:.4f}\n')
                f.write(str(psnr_list)+'\n')
                
            # clear input
            data = []
            
        test_dataset.reset()
    
def static_simulation_model(args, test_dataset):
    for lvl in range(args.level_range[0],args.level_range[1]):
        model = LoadModel(args.task,compression_level=lvl,use_split=args.use_split)
        model.eval()
        img_loss_module = AverageMeter()
        ba_loss_module = AverageMeter()
        psnr_module = AverageMeter()
        all_loss_module = AverageMeter()
        compt_module = AverageMeter()
        decompt_module = AverageMeter()
        decompt_list = []
        video_bpp_module = AverageMeter()
        ds_size = len(test_dataset)
        GoP = args.fP + args.bP +1
        data = []
        all_psnr_list = []
        test_iter = tqdm(range(ds_size))
        eof = False
        for data_idx,_ in enumerate(test_iter):
            if args.evolve and (data_idx == 0 or eof):
                state_list = evolve(model, test_dataset, data_idx, ds_size)
                with open(f'{args.task}.evo.log','a') as f:
                    f.write(str(state_list)+'\n')
            frame,eof = test_dataset[data_idx]
            data.append(transforms.ToTensor()(frame))
            if len(data) < GoP and not eof:
                continue
                
            with torch.no_grad():
                data = torch.stack(data, dim=0)
                if args.use_cuda:
                    data = data.cuda()
                else:
                    data = data.cpu()
                l = data.size(0)
                
                # compress GoP
                com_imgs,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux_loss2,_,_ = parallel_compression(args,model,data,True,lvl)
                ba_loss_module.update(be_loss, l)
                psnr_module.update(psnr,l)
                encoding_time = decoding_time = 0

                compt_module.update(encoding_time,l)
                decompt_module.update(decoding_time,l)
                video_bpp_module.update(be_loss,l)

                decompt_list += [decoding_time]
                decompt_mean = np.array(decompt_list).mean()
                decompt_std = np.array(decompt_list).std()
            
            # show result
            test_iter.set_description(
                f"{data_idx:6}. "
                f"BA: {ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
                f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
                f"E: {compt_module.avg:.3f} D: {decompt_mean:.5f} ({decompt_std:.5f}). ")
                
            # clear input
            data = []

            if eof:
                with open(f'{args.task}.{int(args.evolve)}.log','a') as f:
                    f.write(f'{lvl},{video_bpp_module.avg:.4f},{compt_module.avg:.3f},{decompt_module.avg:.3f}\n')
                    f.write(str(all_psnr_list)+'\n')
                all_psnr_list = []
                compt_module.reset()
                decompt_module.reset()
                video_bpp_module.reset()
                if args.evolve:
                    model = LoadModel(args.task,compression_level=lvl,use_split=args.use_split)
            
        test_dataset.reset()
    return [ba_loss_module.avg,psnr_module.avg]
            

def evolve(model, test_dataset, start, end):
    # should check if evolved version is available
    # if not, training will keep the best version for this video
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    GoP = args.fP + args.bP +1
    min_loss = 100
    max_iter = 30
    max_converge = 3
    max_shrink = 2
    state_list = []
    for encoder_name in ['motion','res']:
        parameters = [p for n, p in model.named_parameters() if (encoder_name+"_encoder") in n]
        # this learning rate to avoid overfitting
        optimizer = torch.optim.Adam([{'params': parameters}], lr=1e-4, weight_decay=5e-4)
        converge_count = shrink_count = 0
        for it in range(max_iter):
            for mode in ['Evo','Test']:
                img_loss_module = AverageMeter()
                ba_loss_module = AverageMeter()
                psnr_module = AverageMeter()
                all_loss_module = AverageMeter()
                data = []
                test_iter = tqdm(range(start, end))
                for _,data_idx in enumerate(test_iter):
                    frame,eof = test_dataset[data_idx]
                    data.append(transforms.ToTensor()(frame))
                    if len(data) < GoP and not eof:
                        continue
                        
                    data = torch.stack(data, dim=0).cuda(device)
                    l = data.size(0)
                    
                    # compress GoP
                    com_imgs,loss,img_loss,be_loss,be_res_loss,psnr,I_psnr,aux_loss,aux_loss2,_,_ = parallel_compression(args,model,data,True,level)
                    ba_loss_module.update(be_loss, l)
                    psnr_module.update(psnr,l)
                    all_loss_module.update(loss.cpu().data.item() if loss else loss,l-1)
                    img_loss_module.update(img_loss,l-1)

                    # backward
                    if mode == 'Evo' and loss:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                            
                    # show result
                    test_iter.set_description(
                        f"{encoder_name} {mode} {data_idx:6} {converge_count} {shrink_count}. "
                        f"B:{ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
                        f"P:{psnr_module.val:.4f} ({psnr_module.avg:.4f}). "
                        f"L:{all_loss_module.val:.4f} ({all_loss_module.avg:.4f}). "
                        f"IL:{img_loss_module.val:.4f} ({img_loss_module.avg:.4f}). ")
                        
                    # clear input
                    data = []

                    if eof:
                        test_dataset._frame_counter = -1
                        break

                if mode == 'Evo':
                    if all_loss_module.avg < min_loss:
                        min_loss = all_loss_module.avg
                        best_state_dict = model.state_dict()
                        converge_count = 0
                    else:
                        converge_count += 1
                        if converge_count == max_converge:
                            if shrink_count < max_shrink:
                                shrink_learning_rate(optimizer)
                                converge_count = 0
                                shrink_count += 1
                            else:
                                break
                else:
                    # record evolution history
                    state_list.append([encoder_name,it,ba_loss_module.avg,psnr_module.avg])

    model.load_state_dict(best_state_dict)
    model.eval()
    return state_list

def shrink_learning_rate(optimizer):
    LR_DECAY_RATE = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= LR_DECAY_RATE

# sudo ufw allow 53
# sudo ufw status verbose



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--role', type=str, default='standalone', help='server or client or standalone')
    parser.add_argument('--dataset', type=str, default='UVG', help='UVG or MCL-JCV')
    parser.add_argument('--task', type=str, default='x264-veryfast', help='RLVC,DVC,SPVC,AE3D,x265,x264')
    parser.add_argument('--use_split', dest='use_split', action='store_true')
    parser.add_argument('--no-use_split', dest='use_split', action='store_false')
    parser.set_defaults(use_split=False)
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    parser.add_argument("--fP", type=int, default=15, help="The number of forward P frames")
    parser.add_argument("--bP", type=int, default=0, help="The number of backward P frames")
    parser.add_argument("--width", type=int, default=2048, help="Input width")
    parser.add_argument("--height", type=int, default=1024, help="Input height")
    parser.add_argument('--level_range', type=int, nargs='+', default=[0,8])
    parser.add_argument('--evolve', action='store_true', help='evolve model')
    parser.add_argument('--max_files', default=0, type=int, help="Maximum loaded files")
    args = parser.parse_args()
    
    # check gpu
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        args.use_split = False

    # setup streaming parameters
    test_dataset = VideoDataset('../dataset/'+args.dataset, frame_size=(args.width,args.height), args.max_files)
    
    if 'x26' in args.task:
        static_simulation_x26x(args, test_dataset)
    elif args.task in ['RLVC2','DVC-pretrained','SSF-Official','Base','ELFVC','ELFVC-SP']:
        static_simulation_model(args, test_dataset)
