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

from models import get_codec_model,parallel_compression,compress_whole_video
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only

class VideoDataset(Dataset):
    def __init__(self, root_dir, resolution, max_files=0):
        self._dataset_dir = os.path.join(root_dir)
        self._frame_size = resolution
        # if resolution == 256:
        #     self._frame_size = (256,256)
        # elif resolution == 720:
        #     self._frame_size = (720,1280)
        # elif resolution == 1080:
        #     self._frame_size = (960,1920)
        # elif resolution == 2160:
        #     self._frame_size = (1920,3840)

        self._total_frames = 0 # Storing file names in object 
        
        self.get_file_names(max_files)
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
            if '.yuv' in self.current_file:
                cap = VideoCaptureYUV(self.current_file)
            else:
                cap = cv2.VideoCapture(self.current_file)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            self._clip = []
            while(True):
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
        
    def get_file_names(self, max_files):
        print("[log] Looking for files in", self._dataset_dir)  
        self.__file_names = []
        for fn in os.listdir(self._dataset_dir):
            fn = fn.strip("'")
            if fn.split('.')[-1] in ['mp4','yuv']:
                self.__file_names.append(self._dataset_dir + '/' + fn)
                if max_files > 0 and len(self.__file_names) == max_files:break
        print("[log] Number of files found {}".format(len(self.__file_names)))  
        
    def __len__(self):
        if not self._total_frames:
            self.count_frames()
        return self._total_frames
        
    def count_frames(self):
        # Count total frames 
        self._total_frames = 0
        for file_name in self.__file_names:
            # print(file_name)
            if '.yuv' in file_name:
                cap = VideoCaptureYUV(file_name)
            else:
                cap = cv2.VideoCapture(file_name)
            # Check if camera opened successfully
            if (cap.isOpened()== False):
                print("Error opening video stream or file")
            # Read until video is completed
            while(True):
                # Capture frame-by-frame
                ret, img = cap.read()
                if ret != True:break
                # cv2.imwrite('../test.jpg',img)
                # exit(0)
                if np.sum(img) == 0:continue
                self._total_frames+=1
            # print(self._total_frames)
            # When everything done, release the video capture object
            cap.release()
        # print("[log] Total frames: ", self._total_frames)
        
class FrameDataset(Dataset):
    def __init__(self, root_dir, frame_size=None):
        self._dataset_dir = os.path.join(root_dir,'vimeo_septuplet','sequences')
        self._train_list_dir = os.path.join(root_dir,'vimeo_septuplet','sep_trainlist.txt')
        self._test_list_dir = os.path.join(root_dir,'vimeo_septuplet','sep_testlist.txt')
        self._frame_size = frame_size
        self._total_frames = 0 # Storing file names in object
        self.get_septuplet_names()
        
    def get_septuplet_names(self):
        print("[log] Looking for septuplets in", self._dataset_dir) 
        self.__septuplet_names = []
        with open(self._train_list_dir,'r') as f:
            for line in f:
                line = line.strip()
                self.__septuplet_names += [self._dataset_dir + '/' + line]
        with open(self._test_list_dir,'r') as f:
            for line in f:
                line = line.strip()
                self.__septuplet_names += [self._dataset_dir + '/' + line]
        print("[log] Number of septuplets found {}".format(len(self.__septuplet_names)))
                
    def __len__(self):
        return len(self.__septuplet_names)
        
    def __getitem__(self, idx):
        data = []
        for img_idx in range(1,8):
            base_dir = self.__septuplet_names[idx]
            img_dir = base_dir+'/'+f'im{img_idx}.png'
            img = Image.open(img_dir).convert('RGB')
            if self._frame_size is not None:
                if img_idx == 1:
                    i, j, h, w = transforms.RandomResizedCrop.get_params(img, (0.08, 1.0), (0.75, 1.3333333333333333))
                img = transforms.functional.resized_crop(img, i, j, h, w,(self._frame_size,self._frame_size))
                # img = img.resize((self._frame_size,self._frame_size)) 
            data.append(transforms.ToTensor()(img))
        data = torch.stack(data, dim=0)
        return data

categories = ['lobby','retail','office','industry_safety','cafe_shop']
num_views = [4,6,5,4,4]
class MultiViewVideoDataset(Dataset):
    def __init__(self, root_dir, category_id=0, split='test', gop_size=16):
        self._dataset_dir = os.path.join(root_dir)
        self._dirs = []
        self._dirs += [os.path.join(root_dir,'train','images','63am')]
        self._dirs += [os.path.join(root_dir,'train','images','64am')]
        self._dirs += [os.path.join(root_dir,'validation','images','64pm')]
        self.category = categories[category_id]
        self.num_views = num_views[category_id]
        self.split = split
        self.gop_size = gop_size
        
        self.get_file_names()
        
    def get_file_names(self):
        print("[log] Looking for files in", self._dataset_dir)  
        self.__file_names = []
        self.__video_frames = []
        self.__video_gops = []
        for directory in self._dirs:
            for fn in os.listdir(directory):
                if self.category in fn:
                    fn = fn.strip("'")
                    if '_0' in fn and self.split == 'train':continue
                    if '_0' not in fn and self.split == 'test':continue
                    self.__file_names += [os.path.join(directory,fn)]
                    self.__video_frames += [len(os.listdir(os.path.join(directory,fn)))//self.num_view]
                    self.__video_gops += [len(os.listdir(os.path.join(directory,fn)))//self.num_view//self.gop_size]
        self.__num_gops = sum(self.__video_gops)
        print(self.__file_names)
        print("[log] Number of files found {}".format(len(self.__file_names)))  
        exit(0)

    def __len__(self):
        return len(self.__num_gops)
        
    def __getitem__(self, idx):
        file_idx = 0
        total_gops = 0
        for gops in self.__video_gops:
            gop_idx = idx - total_gops
            if gop_idx < gops:
                break
            total_gops += gops
            file_idx += 1

        data = []
        for v in range(self.num_views):
            for g in range(self.gop_size):
                frame_idx = gop_idx * self.gop_size + g
                img_dir = os.path.join(self.__file_names[file_idx],f'rgb_{frame_idx:05d}_{v+1}.jpg')
                img = Image.open(img_dir).convert('RGB')
                data.append(transforms.ToTensor()(img))
        data = torch.stack(data, dim=0)
        data = data.view(self.num_views,self.gop_size,3,data.size(2),data.size(3))
        return data