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
                        
def static_simulation_x26x(test_dataset,name='x264'):
    print('Benchmarking:',name)
    ds_size = len(test_dataset)
    
    for Q in [15,19,23,27]:
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
                
            psnr_list,msssim_list,bpp_act_list = compress_whole_video(name,data,Q,*test_dataset._frame_size)
            
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
    
def static_bench_x26x():
    # optionaly try x264,x265
    test_dataset = VideoDataset('../dataset/MCL-JCV', frame_size=(256,256))
    static_simulation_x26x(test_dataset,'x264')
    static_simulation_x26x(test_dataset,'x265')
    test_dataset = VideoDataset('../dataset/UVG', frame_size=(256,256))
    static_simulation_x26x(test_dataset,'x264')
    static_simulation_x26x(test_dataset,'x265')
    exit(0)

def dynamic_simulation_x26x(test_dataset, name='x264'):
    print('Benchmarking:',name)
    ds_size = len(test_dataset)
    
    def create_client(Q,width=256,height=256):
        fps = 25
        GOP = 13
        if name == 'x265':
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p '+\
                    f'-preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" '+\
                    f'-rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8555/live'
        elif name == 'x264':
            cmd = f'/usr/bin/ffmpeg -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p '+\
                    f'-preset veryfast -tune zerolatency -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug '+\
                    f'-rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8555/live'
        else:
            print('Codec not supported')
            exit(1)
        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
        return process
        
    def client(raw_clip,process):
        for idx,img in enumerate(raw_clip):
            img = np.array(img)
            process.stdin.write(img.tobytes())
            print('write',idx)
            #time.sleep(1/60.)
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()
        
    def create_server():
        # serve as a rtsp server
        # ssh -R [REMOTE:]REMOTE_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
        # ssh -R 8555:localhost:8555 uiuc@192.168.251.195
        command = ['/usr/bin/ffmpeg',
            '-rtsp_flags', 'listen',
            '-i', 'rtsp://127.0.0.1:8555/live?tcp?',
            '-f', 'image2pipe',    # Use image2pipe demuxer
            '-pix_fmt', 'bgr24',   # Set BGR pixel format
            '-vcodec', 'rawvideo', # Get rawvideo output format.
            '-']
            
        # Open sub-process that gets in_stream as input and uses stdout as an output PIPE.
        p_server = sp.Popen(command, stdout=sp.PIPE)
        
        return p_server
        
    # how to direct rtsp traffic?
    def server(data,Q,width=256,height=256):
        # Beginning time of streaming
        t_0 = time.perf_counter()
        
        # create a rtsp server or listener
        p_server = create_server()
        print('server created')
        
        # create a rtsp track
        p_client = create_client(Q)
        print('client created')
        
        # Start a thread that streams data
        threading.Thread(target=client, args=(data,p_client,)).start() 
        
        psnr_list = []
        msssim_list = []
        
        i = 0
        
        t_warmup = None
        
        stream_iter = tqdm(range(len(data)))
        while True:
            # read width*height*3 bytes from stdout (1 frame)
            raw_frame = p_server.stdout.read(width*height*3)
            if t_warmup is None:
                t_warmup = time.perf_counter() - t_0
                print('Warm-up:',t_warmup)

            if len(raw_frame) != (width*height*3):
                #print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
                break

            # Convert the bytes read into a NumPy array, and reshape it to video frame dimensions
            frame = np.fromstring(raw_frame, np.uint8)
            frame = frame.reshape((height, width, 3))
            
            # process metrics
            com = transforms.ToTensor()(frame).cuda().unsqueeze(0)
            raw = transforms.ToTensor()(data[i]).cuda().unsqueeze(0)
            psnr_list += [PSNR(raw, com)]
            msssim_list += [MSSSIM(raw, com)]
            i += 1
            
            # Count time
            total_time = time.perf_counter() - t_0
            fps = i/total_time
        
            # show result
            stream_iter.set_description(
                f"{i:3}. "
                f"FPS: {fps:.2f}. "
                f"PSNR: {float(psnr_list[-1]):.2f}. "
                f"MSSSIM: {float(msssim_list[-1]):.4f}. "
                f"Total: {total_time:.3f}. ")
        return psnr_list,msssim_list
            
    from collections import deque
    
    for Q in [15,19,23,27]:
        data = []
        psnr_module = AverageMeter()
        msssim_module = AverageMeter()
        test_iter = tqdm(range(ds_size))
        for data_idx,_ in enumerate(test_iter):
            frame,eof = test_dataset[data_idx]
            data.append(frame)
            if not eof:
                continue
            l = len(data)
            
            psnr_list, msssim_list = server(data,Q)
                
            # aggregate loss
            psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
            msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
            
            # record loss
            psnr_module.update(psnr.cpu().data.item(),l)
            msssim_module.update(msssim.cpu().data.item(), l)
            
            # show result
            test_iter.set_description(
                f"{data_idx:6}. "
                f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
                f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). ")
                
            # clear input
            data = []
            exit(0)
            
        test_dataset.reset()

def LoadModel(CODEC_NAME):
    #CODEC_NAME = 'SPVC-stream'
    loss_type = 'P'
    compression_level = 2
    RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth'

    ####### Codec model 
    model = get_codec_model(CODEC_NAME,noMeasure=False,loss_type=loss_type,compression_level=compression_level)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    ####### Load codec model 
    if os.path.isfile(RESUME_CODEC_PATH):
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
    return model
    
def streaming_parallel(model, test_dataset):
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
        if not eof: continue
        data = torch.stack(data, dim=0).cuda()
        L = data.size(0)
        def client(data):
            # start a process to pipe data to netcat
            cmd = f'nc localhost 8888'
            process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
            for begin in range(0,L,GoP):
                # put a timer here
                with torch.no_grad():
                    x_GoP = data[begin:begin+GoP]
                    if x_GoP.size(0)>fP+1:
                        # compress I
                        # compress backward
                        x_b = torch.flip(x_GoP[:fP+1],[0])
                        mv_string1,res_string1,_ = model.compress(x_GoP)
                        # compress forward
                        x_f = data[fP:]
                        mv_string2,res_string2,_ = model.compress(x_f)
                        com_data = [x_GoP[:1],mv_string1,res_string1,mv_string2,res_string2]
                    else:
                        # compress I
                        # compress backward
                        x_b = torch.flip(data,[0])
                        mv_string,res_string,bpp_act_list = model.compress(x_b)
                        com_data = [x_GoP[:1],mv_string,res_string]
                assert(len(com_data[0])==2 and len(com_data[1])==2)
                # Send compressed I frame (todo)
                # [B=1] check number of elements to send
                bytes_send = struct.pack('B',len(com_data[1:]))
                process.stdin.write(bytes_send)
                # send compressed strings
                for x_string,z_string in com_data[1:]:
                    # [L=8] send length of next string
                    x_len = len(x_string)
                    bytes_send = struct.pack('L',x_len)
                    process.stdin.write(bytes_send)
                    # send actual string
                    process.stdin.write(x_string)
                    # [L=8] send length of next string
                    z_len = len(z_string)
                    bytes_send = struct.pack('L',z_len)
                    process.stdin.write(bytes_send)
                    # send actual string
                    process.stdin.write(z_string)
            # Close and flush stdin
            process.stdin.close()
            # Wait for sub-process to finish
            process.wait()
            # Terminate the sub-process
            process.terminate()
        
        def server(data):
            # create a pipe for listening from netcat
            cmd = f'nc -l 8888'
            process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
            # Start a thread that streams data
            threading.Thread(target=client, args=(data,)).start() 
            # initialize
            psnr_list = []
            msssim_list = []
            i = 0
            stream_iter = tqdm(range(L))
            # start listening
            for begin in range(0,L,GoP):
                # decompress I frame
                com_data = [data[begin:begin+1]]
                # [B=1] receive number of elements
                bytes_recv = process.stdout.read(1)
                num_of_elem = struct.unpack('B',bytes_recv)
                # receive compressed strings
                for _ in range(num_of_elem):
                    # [L=8] receive length of next string
                    bytes_recv = process.stdout.read(8)
                    x_len = struct.unpack('L',bytes_recv)
                    # recv actual string
                    x_string = process.stdout.read(x_len)
                    # [L=8] receive length of next string
                    bytes_recv = process.stdout.read(8)
                    z_len = struct.unpack('L',bytes_recv)
                    # recv actual string
                    z_string = process.stdout.read(z_len)
                    com_data += [(x_string,z_string)]
                with torch.no_grad():
                    if len(com_data)==5:
                        # decompress I
                        x_ref,mv_string1,res_string1,mv_string2,res_string2 = com_data
                        # decompress backward
                        x_b_hat = model.decompress(x_ref,mv_string1,res_string1)
                        # decompress forward
                        x_f_hat = model.decompress(x_ref,mv_string2,res_string2)
                        # concate
                        x_hat = torch.cat((torch.flip(x_b_hat,[0]),x_ref,x_f_hat),dim=0)
                    else:
                        # decompress I
                        x_ref,mv_string,res_string = com_data
                        # decompress backward
                        x_b_hat = model.decompress(x_ref,mv_string,res_string)
                        # concate
                        x_hat = torch.cat((torch.flip(x_b_hat,[0]),x_ref),dim=0)
                    for com in x_hat:
                        com = com.cuda().unsqueeze(0)
                        raw = data[i].cuda().unsqueeze(0)
                        psnr_list += [PSNR(raw, com)]
                        msssim_list += [MSSSIM(raw, com)]
                        i += 1
                        # show result
                        stream_iter.set_description(
                            f"{i:3}. "
                            f"PSNR: {float(psnr_list[-1]):.2f}. "
                            f"MSSSIM: {float(msssim_list[-1]):.4f}. ")
            # Close and flush stdin
            process.stdout.close()
            # Wait for sub-process to finish
            process.wait()
            # Terminate the sub-process
            process.terminate()
            print('close server')
            return psnr_list,msssim_list
                
        psnr_list,msssim_list = server(data)
            
        # aggregate loss
        psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
        msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
        
        # record loss
        psnr_module.update(psnr.cpu().data.item(),l)
        msssim_module.update(msssim.cpu().data.item(), l)
        
        # show result
        test_iter.set_description(
            f"{data_idx:6}. "
            f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). ")
            
        # clear input
        data = []
        
    test_dataset.reset()
    
def streaming_sequential(model, test_dataset, use_gpu=True):
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
        if not eof: continue
            
        data = torch.stack(data, dim=0)
        if use_gpu:
            data = data.cuda()
        L = data.size(0)
        # put a timer for backward frames
        # a different timer for forward frames
        proc_iter = tqdm(range(0,L,GoP))
        for _,begin in enumerate(proc_iter):
            with torch.no_grad():
                x_GoP = data[begin:begin+GoP]
                if x_GoP.size(0)>fP+1:
                    # compress I
                    # compress backward
                    x_b = torch.flip(x_GoP[:fP+1],[0])
                    B,_,H,W = x_b.size()
                    com_hidden = model.init_hidden(H,W)
                    com_mv_prior_latent = com_res_prior_latent = None
                    decom_hidden = model.init_hidden(H,W)
                    decom_mv_prior_latent = decom_res_prior_latent = None
                    x_ref = x_b[0:1]
                    # wait until the group is ready (e.g., count for 6*0.03), then compress
                    # make sure the fetch interval of two frames is at least 1/60. or 1/30.
                    psnr_list1 = []
                    msssim_list1 = []
                    bpp_act_list1 = []
                    for i in range(1,B):
                        # need to enable compress and decompress to have separate prior_latent!!!!!!!!!!!!
                        mv_string,res_string,bpp_act,com_hidden,mv_size,res_size,com_mv_prior_latent,com_res_prior_latent = \
                            model.compress(x_ref, x_b[i:i+1], com_hidden, i>1, com_mv_prior_latent, com_res_prior_latent)
                        x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                            model.decompress(x_ref, mv_string, res_string, decom_hidden, i>1, mv_size, res_size, decom_mv_prior_latent, decom_res_prior_latent)
                        x_ref = x_ref.detach()
                        raw = x_b[i:i+1]
                        psnr_list1 += [PSNR(raw, x_ref)]
                        msssim_list1 += [MSSSIM(raw, x_ref)]
                        bpp_act_list1 += [bpp_act]
                    
                    # compress forward
                    x_f = x_GoP[fP:]
                    # compress as soon as a new frame is ready
                    B,_,H,W = x_f.size()
                    com_hidden = model.init_hidden(H,W)
                    com_mv_prior_latent = com_res_prior_latent = None
                    decom_hidden = model.init_hidden(H,W)
                    decom_mv_prior_latent = decom_res_prior_latent = None
                    x_ref = x_f[0:1]
                    psnr_list2 = []
                    msssim_list2 = []
                    bpp_act_list2 = []
                    for i in range(1,B):
                        mv_string,res_string,bpp_act,com_hidden,mv_size,res_size,com_mv_prior_latent,com_res_prior_latent = \
                            model.compress(x_ref, x_f[i:i+1], com_hidden, i>1, com_mv_prior_latent, com_res_prior_latent)
                        x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                            model.decompress(x_ref, mv_string, res_string, decom_hidden, i>1, mv_size, res_size, decom_mv_prior_latent, decom_res_prior_latent)
                        x_ref = x_ref.detach()
                        raw = x_f[i:i+1]
                        psnr_list2 += [PSNR(raw, x_ref)]
                        msssim_list2 += [MSSSIM(raw, x_ref)]
                        bpp_act_list2 += [bpp_act]
                    # concat 
                    psnr_list = psnr_list1[::-1] + [torch.FloatTensor([40]).squeeze(0).to(data.device)] + psnr_list2
                    msssim_list = msssim_list1[::-1] + [torch.FloatTensor([1]).squeeze(0).to(data.device)] + msssim_list2
                    bpp_act_list = bpp_act_list1[::-1] + [torch.FloatTensor([1]).squeeze(0)] + bpp_act_list2
                else:
                    # compress I
                    # compress forward
                    x_b = torch.flip(x_GoP,[0])
                    psnr_list = []
                    msssim_list = []
                    bpp_act_list = []
                    for i in range(1,B):
                        mv_string,res_string,bpp_act,_,mv_size,res_size = model.compress(x_ref, x_b[i:i+1], hidden, i>1)
                        com,hidden = model.decompress(x_ref, mv_string, res_string, hidden, i>1, mv_size, res_size)
                        raw = x_b[i:i+1]
                        psnr_list += [PSNR(raw, com)]
                        msssim_list += [MSSSIM(raw, com)]
                        bpp_act_list += [bpp_act]
                        x_ref = com.detach()
                    # concat 
                    psnr_list = psnr_list[::-1] + [torch.FloatTensor([40]).squeeze(0).to(data.device)]
                    msssim_list = msssim_list[::-1] + [torch.FloatTensor([1]).squeeze(0).to(data.device)]
                    bpp_act_list = bpp_act_list[::-1] + [torch.FloatTensor([1]).squeeze(0)]
            
            # aggregate loss
            ba_loss = torch.stack(bpp_act_list,dim=0).mean(dim=0)
            psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
            msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
            
            # record loss
            ba_loss_module.update(ba_loss.cpu().data.item(), len(psnr_list))
            psnr_module.update(psnr.cpu().data.item(),len(psnr_list))
            msssim_module.update(msssim.cpu().data.item(), len(psnr_list))
        
            # show result
            proc_iter.set_description(
                f"{data_idx:6}. "
                f"BA: {ba_loss_module.val:.2f} ({ba_loss_module.avg:.2f}). "
                f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
                f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). "
                f"I: {float(max(psnr_list)):.2f}")
        
        # clear input
        data = []
        
    test_dataset.reset()
    
def static_simulation_model(model, test_dataset):
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
# todo: a protocol to send strings of compressed frames
# complete I frame comrpession
# complete the compression/decompress function for DVC and RLVC
# complete a streaming sequential function
# run this script in docker
# then test throughput(fps) and rate-distortion on different devices and different losses

        
####### Load dataset
test_dataset = VideoDataset('../dataset/UVG', frame_size=(256,256))
####### Load model
model = LoadModel('SPVC')

# try x265,x264 streaming with Gstreamer
#dynamic_simulation_x26x(test_dataset, 'x264')
streaming_parallel(model, test_dataset)
#static_simulation_model(model, test_dataset)
#streaming_sequential(model, test_dataset)
enc,dec = showTimer(model)