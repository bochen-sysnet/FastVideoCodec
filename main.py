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

from models import get_codec_model,parallel_compression,update_training
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

class VideoDataset(Dataset):
    def __init__(self, root_dir, frame_size=(256,256)):
        self._dataset_dir = os.path.join(root_dir)
        
        self.get_file_names() # Storing file names in object 
        
        self._image = None 
        self._num_files = len(self.__file_names)
        self._curr_counter = 0
        self._num_frames = 0
        self._total_frames = 0
        self._sample_list = []
        self._frame_counter = -1 # Count the number of frames used per file
        self._file_counter = -1 # Count the number of files used
        self._dataset_nums = [] # Number of frames to be considered from each file (records+files)
        self._dataset_itr = None # tfRecord iterator
        self.num_sample = self._total_frames
        self._clip = [] # hold video frames
        self._frame_size = frame_size
        
    @property
    def data(self):
        self._curr_counter+=1
        return self.__getitem__(self._curr_counter)
        
    def __getitem__(self, idx):
        # Get the next dataset if frame number is more than table count
        if not len(self._dataset_nums) or self._frame_counter >= self._dataset_nums[self._file_counter]-1: 
            self.current_file = self.__file_names.pop() # get one filename
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
                img = img.resize(self._frame_size)
                self._clip.append(transforms.ToTensor()(img))
            self._file_counter +=1
            self._dataset_nums.append(len(self._clip))
            self._frame_counter = 0
        else:
            self._frame_counter+=1
        self._num_frames+=1
        return self._clip[self._frame_counter],self._frame_counter==self._dataset_nums[self._file_counter]-1
        
    def get_file_names(self):
        print("[log] Looking for files in", self._dataset_dir)  
        self.__file_names = []
        for fn in os.listdir(self._dataset_dir):
            fn = fn.strip("'")
            if fn.split('.')[-1] == 'mp4':
                self.__file_names.append(self._dataset_dir + '/' + fn)
            # test with only 5 files
            if len(self.__file_names)==1:break 
        print("[log] Number of files found {}".format(len(self.__file_names)))  
        
    def __len__(self):
        if not self._total_frames:
            self.count_frames()
        return self._total_frames
        
    def count_frames(self):
        # Count total frames 
        self._total_frames = 0
        for file_name in self.__file_names:
            print(file_name)
            cap = cv2.VideoCapture(file_name)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
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

# OPTION
BACKUP_DIR = '/home/monet/research/FastVideoCodec/backup'
CODEC_NAME = 'SPVC'
RESUME_CODEC_PATH = '/home/monet/research/YOWO/backup/ucf24/yowo_ucf24_16f_SPVC-P_best.pth'
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 5e-4
BEGIN_EPOCH = 1
END_EPOCH = 10

####### Check backup directory, create if necessary
# ---------------------------------------------------------------
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)
    
####### Load dataset
train_dataset = VideoDataset('../dataset/vimeo', frame_size=(256,256))

####### Create model
seed = int(time.time())
#seed = int(0)
torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # TODO: add to config e.g. 0,1,2,3
    torch.cuda.manual_seed(seed)

# codec model .
model = get_codec_model(CODEC_NAME)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

####### Create optimizer
# ---------------------------------------------------------------
parameters = [p for n, p in model.named_parameters() if (not n.endswith(".quantiles"))]
aux_parameters = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
optimizer = torch.optim.Adam([{'params': parameters},{'params': aux_parameters, 'lr': 10*LEARNING_RATE}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# initialize best score
best_score = 0 
best_codec_score = [0,1]
score = [0,1]

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
    best_codec_score = checkpoint['score']
    load_state_dict_all(model, checkpoint['state_dict'])
    print("Loaded model codec score: ", checkpoint['score'])
    del checkpoint
else:
    print("Cannot load model codec", CODEC_NAME)
print("===================================================================")
        
def train(epoch, model, train_dataset, optimizer):
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
    
    train_iter = tqdm(range(ds_size))
    data = []
    for data_idx,_ in enumerate(train_iter):
        for j in range(batch_size):
            frame,eof = train_dataset[data_idx]
            data.append(frame)
            if eof:break
        data = torch.stack(data, dim=0).cuda()
        l = data.size(0)-1
        
        # run model
        img_loss_list,bpp_est_list,aux_loss_list,psnr_list,msssim_list,_ = parallel_compression(model,data)
        
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
        if eof:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # clear input
        data = []
    
def test(epoch, model, test_dataset):
    pass

def save_checkpoint(state, is_best, directory, CODEC_NAME):
    import shutil
    torch.save(state, '%s/%s_checkpoint.pth' % (directory, CODEC_NAME))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (directory, CODEC_NAME),
                        '%s/%s_best.pth' % (directory, CODEC_NAME))
                        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    LEARNING_RATE = 1e-4
    LR_DECAY_RATE = 0.1
    STEPS = []
    lr_new = LEARNING_RATE * (LR_DECAY_RATE ** (sum(epoch >= np.array(STEPS))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
    return lr_new

for epoch in range(BEGIN_EPOCH, END_EPOCH + 1):
    # Adjust learning rate
    r = adjust_learning_rate(optimizer, epoch)
    
    # Train and test model
    print('training at epoch %d, r=%.2f' % (epoch,r))
    train(epoch, model, train_dataset, optimizer)
    
    #print('testing at epoch %d' % (epoch))
    #score = test(epoch, model, test_dataset)

    state = {
        'epoch': epoch,
        'state_dict': model_codec.state_dict(),
        'score': score
        }
    save_checkpoint(state, True, BACKUP_DIR, CODEC_NAME)
    print('Weights are saved to backup directory: %s' % (BACKUP_DIR))