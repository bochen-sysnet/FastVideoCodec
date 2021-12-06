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

from models import get_codec_model,parallel_compression,update_training,compress_whole_video,showTimer
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only
from models import PSNR,MSSSIM

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
            break
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
        #print("[log] Total frames: ", self._total_frames)

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

def save_checkpoint(state, is_best, directory, CODEC_NAME):
    import shutil
    torch.save(state, f'{directory}/{CODEC_NAME}/{CODEC_NAME}-1024P_ckpt.pth')
    if is_best:
        shutil.copyfile(f'{directory}/{CODEC_NAME}/{CODEC_NAME}-1024P_ckpt.pth',
                        f'{directory}/{CODEC_NAME}/{CODEC_NAME}-1024P_best.pth')

def test_x26x(name='x264'):
    print('Benchmarking:',name)
    out_send = cv2.VideoWriter(\
        'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=8888',\
        cv2.CAP_GSTREAMER,0, 20, (1024,512), True)
    for f in range(100):
        imarray = np.random.rand(100,100,3) * 255
    

# try x265,x264 streaming with Gstreamer
test_x26x('x264')
exit(0)
        
# OPTION
BACKUP_DIR = 'backup'
CODEC_NAME = 'SPVC-P'
RESUME_CODEC_PATH = f'backup/SPVC/SPVC-1024P_best.pth'
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
BEGIN_EPOCH = 1
END_EPOCH = 10

####### Check backup directory, create if necessary
# ---------------------------------------------------------------
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

####### Create model
seed = int(time.time())
#seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

# codec model .
model = get_codec_model(CODEC_NAME,noMeasure=False)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

####### Create optimizer
# ---------------------------------------------------------------
parameters = [p for n, p in model.named_parameters() if (not n.endswith(".quantiles"))]
aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
optimizer = torch.optim.Adam([{'params': parameters},{'params': aux_parameters, 'lr': 10*LEARNING_RATE}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# initialize best score
best_score = 0 
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
    pretrained_model_path = "/home/monet/research/YOWO/backup/ucf24/yowo_ucf24_16f_SPVC_best.pth"
    checkpoint = torch.load(pretrained_model_path)
    load_state_dict_whatever(model_codec, checkpoint['state_dict'])
    del checkpoint
elif RESUME_CODEC_PATH and os.path.isfile(RESUME_CODEC_PATH):
    print("Loading for ", CODEC_NAME, 'from',RESUME_CODEC_PATH)
    checkpoint = torch.load(RESUME_CODEC_PATH)
    BEGIN_EPOCH = checkpoint['epoch'] + 1
    best_codec_score = checkpoint['score'][1:4]
    load_state_dict_all(model, checkpoint['state_dict'])
    print("Loaded model codec score: ", checkpoint['score'])
    del checkpoint
else:
    print("Cannot load model codec", CODEC_NAME)
    exit(1)
print("===================================================================")
    
def streaming(model, test_dataset):
    ba_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    msssim_module = AverageMeter()
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
            # need to have sequential and batch streaming
            # video will come at different rates 30-60fps
            # network will have different bandwidth
            # unlimited rate?
            if l>fP+1:
                # compress backward
                x_raw = torch.flip(data[:fP+1],[0])
                mv_string,res_string,bpp_act_list1 = model.compress(x_raw)
                x_hat = model.decompress(x_raw[:1], mv_string,res_string)
                psnr_list1 = PSNR(x_raw[1:], x_hat, use_list=True)
                msssim_list1 = MSSSIM(x_raw[1:], x_hat, use_list=True)
                # compress forward
                x_raw = data[fP:]
                mv_string,res_string,bpp_act_list2 = model.compress(x_raw)
                x_hat = model.decompress(x_raw[:1], mv_string,res_string)
                psnr_list2 = PSNR(x_raw[1:], x_hat, use_list=True)
                msssim_list2 = MSSSIM(x_raw[1:], x_hat, use_list=True)
                # concate
                psnr_list = psnr_list1[::-1] + [torch.FloatTensor([40]).squeeze(0).cuda()] + psnr_list2
                msssim_list = msssim_list1[::-1] + [torch.FloatTensor([1]).squeeze(0).cuda()] + msssim_list2
                bpp_act_list = bpp_act_list1[::-1] + [torch.FloatTensor([1]).squeeze(0).cuda()] + bpp_act_list2
            else:
                # compress backward
                x_raw = torch.flip(data,[0])
                mv_string,res_string,bpp_act_list = model.compress(x_raw)
                x_hat = model.decompress(x_raw[:1], mv_string,res_string)
                psnr_list = PSNR(x_raw[1:], x_hat, use_list=True)
                msssim_list = MSSSIM(x_raw[1:], x_hat, use_list=True)
                # concate
                psnr_list =  psnr_list[::-1] + [torch.FloatTensor([40]).squeeze(0).cuda()]
                msssim_list = msssim_list[::-1] + [torch.FloatTensor([1]).squeeze(0).cuda()]
                bpp_act_list = bpp_act_list[::-1] + [torch.FloatTensor([1]).squeeze(0).cuda()]
                
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
            f"{data_idx:6}. "
            f"BA: {ba_loss_module.val:.2f} ({ba_loss_module.avg:.2f}). "
            f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). ")
            
        # clear input
        data = []
        
    test_dataset.reset()
        
####### Load dataset
test_dataset = VideoDataset('../dataset/UVG', frame_size=(256,256))

# Train and test model
streaming(model, test_dataset)
enc,dec = showTimer(model)