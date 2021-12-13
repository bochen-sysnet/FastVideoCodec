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

from models import get_codec_model,parallel_compression,update_training,compress_whole_video,showTimer
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only
from models import PSNR,MSSSIM

def LoadModel(CODEC_NAME,compression_level = 2,use_cuda=True):
    loss_type = 'P'
    if CODEC_NAME == 'SPVC-stream':
        RESUME_CODEC_PATH = f'backup/SPVC/SPVC-{compression_level}{loss_type}_best.pth'
    else:
        RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth'

    ####### Codec model 
    model = get_codec_model(CODEC_NAME,noMeasure=False,loss_type=loss_type,compression_level=compression_level,use_cuda=use_cuda)
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
                        
def static_simulation_x26x(args,test_dataset):
    ds_size = len(test_dataset)
    
    Q_list = [15,19,23,27] if args.Q_option == 'Slow' else [23]
    for Q in Q_list:
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
                
            psnr_list,msssim_list,bpp_act_list = compress_whole_video(args.task,data,Q,*test_dataset._frame_size)
            
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
    
def static_simulation_model(args, test_dataset):
    aux_loss_module = AverageMeter()
    img_loss_module = AverageMeter()
    ba_loss_module = AverageMeter()
    be_loss_module = AverageMeter()
    psnr_module = AverageMeter()
    msssim_module = AverageMeter()
    all_loss_module = AverageMeter()
    ds_size = len(test_dataset)
    model = LoadModel(args.task)
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

def block_until_open(ip_addr,port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        result = s.connect_ex((ip_addr,int(port)))
        if result == 0:
            #print('Port OPEN')
            break
        else:
            #print('Port CLOSED, connect_ex returned: '+str(result))
            time.sleep(0.1)
    s.close()
    
def x26x_client(args,data,model=None,Q=None,width=256,height=256):
    fps = 25
    GOP = 13
    if args.task == 'x265':
        cmd = f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p '+\
                f'-preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" '+\
                f'-rtsp_transport tcp -f rtsp rtsp://{args.server_ip}:{args.server_port}/live'
    elif args.task == 'x264':
        cmd = f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p '+\
                f'-preset veryfast -tune zerolatency -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug '+\
                f'-rtsp_transport tcp -f rtsp rtsp://{args.server_ip}:{args.server_port}/live'
    else:
        print('Codec not supported')
        exit(1)
    block_until_open(args.server_ip,args.probe_port)
    # create a rtsp track
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    t_0 = None
    for idx,img in enumerate(data):
        # read data
        # wait for 1/30. or 1/60.
        img = np.array(img)
        while t_0 is not None and time.perf_counter() - t_0 < 1/60.:time.sleep(0.01)
        t_0 = time.perf_counter()
        process.stdin.write(img.tobytes())
    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()
    
# how to direct rtsp traffic?
def x26x_server(args,data,model=None,Q=None,width=256,height=256):
    # create a rtsp server or listener
    # ssh -R [REMOTE:]REMOTE_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
    # ssh -R 8555:localhost:8555 uiuc@192.168.251.195
    command = ['/usr/bin/ffmpeg',
        '-hide_banner', '-loglevel', 'error',
        '-rtsp_flags', 'listen',
        '-i', f'rtsp://{args.server_ip}:{args.server_port}/live?tcp?',
        '-f', 'image2pipe',    # Use image2pipe demuxer
        '-pix_fmt', 'bgr24',   # Set BGR pixel format
        '-vcodec', 'rawvideo', # Get rawvideo output format.
        '-']
        
    # Open sub-process that gets in_stream as input and uses stdout as an output PIPE.
    process = sp.Popen(command, stdout=sp.PIPE)
    # Probe port (server port in rtsp cannot be probed)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.server_ip, int(args.probe_port)))
    s.listen(1)
    conn, addr = s.accept()
    # Beginning time of streaming
    t_0 = time.perf_counter()
    
    psnr_list = []
    i = 0
    t_warmup = None
    stream_iter = tqdm(range(len(data)))
    while True:
        # read width*height*3 bytes from stdout (1 frame)
        raw_frame = process.stdout.read(width*height*3)
        if t_warmup is None:
            t_warmup = time.perf_counter() - t_0

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
        i += 1
        
        # Count time
        total_time = time.perf_counter() - t_0
        fps = i/total_time
    
        # show result
        stream_iter.set_description(
            f"{i:3}. "
            f"FPS: {fps:.2f}. "
            f"PSNR: {float(psnr_list[-1]):.2f}. "
            f"Total: {total_time:.3f}. ")
    conn.close()
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_list,fps,t_warmup
    
def send_strings_to_process(process, strings, useXZ=True):
    for string_list in strings:
        if useXZ:
            x_string_list,z_string_list = string_list
        else:
            x_string_list = string_list
        for x_string in x_string_list:
            # [L=8] send length of next string
            x_len = len(x_string)
            bytes_send = struct.pack('L',x_len)
            process.stdin.write(bytes_send)
            # send actual string
            process.stdin.write(x_string)
        if useXZ:
            for z_string in z_string_list:
                # [L=8] send length of next string
                z_len = len(z_string)
                bytes_send = struct.pack('L',z_len)
                process.stdin.write(bytes_send)
                # send actual string
                process.stdin.write(z_string)
    
def recv_strings_from_process(process, strings_to_recv, useXZ=True):
    x_string_list = []
    for _ in range(strings_to_recv):
        # [L=8] receive length of next string
        bytes_recv = process.stdout.read(8)
        x_len = struct.unpack('L',bytes_recv)[0]
        # recv actual string
        x_string = process.stdout.read(x_len)
        x_string_list += [x_string]
    if useXZ:
        z_string_list = []
        for _ in range(strings_to_recv):
            # [L=8] receive length of next string
            bytes_recv = process.stdout.read(8)
            z_len = struct.unpack('L',bytes_recv)[0]
            # recv actual string
            z_string = process.stdout.read(z_len)
            z_string_list += [z_string]
    strings = (x_string_list,z_string_list) if useXZ else x_string_list
    return strings
            
def SPVC_AE3D_client(args,data,model=None,Q=None,fP=6,bP=6):
    block_until_open(args.server_ip,args.server_port)
    GoP = fP+bP+1
    L = data.size(0)
    # start a process to pipe data to netcat
    cmd = f'nc {args.server_ip} {args.server_port}'
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
    for begin in range(0,L,GoP):
        x_GoP = data[begin:begin+GoP]
        GoP_size = x_GoP.size(0)
        # Send GoP size, this determines how to encode/decode the strings
        bytes_send = struct.pack('B',GoP_size)
        process.stdin.write(bytes_send)
        # Send compressed I frame (todo)
        if GoP_size>fP+1:
            # compress I
            # compress backward
            x_b = torch.flip(x_GoP[:fP+1],[0])
            mv_string1,res_string1,_ = model.compress(x_b)
            # Send strings in order
            send_strings_to_process(process, [mv_string1,res_string1])
            # compress forward
            x_f = x_GoP[fP:]
            mv_string2,res_string2,_ = model.compress(x_f)
            # Send strings in order
            send_strings_to_process(process, [mv_string2,res_string2])
        else:
            # compress I
            # compress backward
            x_f = x_GoP
            mv_string,res_string,bpp_act_list = model.compress(x_f)
            # Send strings in order
            send_strings_to_process(process, [mv_string,res_string])
    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()
    
def SPVC_AE3D_server(args,data,model=None,Q=None,fP=6,bP=6):
    GoP = fP+bP+1
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.server_ip}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    # Beginning time of streaming
    t_0 = time.perf_counter()
    # initialize
    psnr_list = []
    i = 0
    t_warmup = None
    L = data.size(0)
    stream_iter = tqdm(range(L))
    # start listening
    for begin in range(0,L,GoP):
        # decompress I frame
        x_GoP = data[begin:begin+GoP]
        x_ref = x_GoP[fP:fP+1] if x_GoP.size(0)>fP+1 else x_GoP[:1]
        # [B=1] receive number of elements
        bytes_recv = process.stdout.read(1)
        GoP_size = struct.unpack('B',bytes_recv)[0]
        # receive strings based on gop size
        if GoP_size>fP+1:
            # receive the first two strings
            strings_to_recv = fP
            mv_string1 = recv_strings_from_process(process, strings_to_recv)
            res_string1 = recv_strings_from_process(process, strings_to_recv)
            # decompress backward
            x_b_hat = model.decompress(x_ref,mv_string1,res_string1)
            # receive the second two strings
            strings_to_recv = GoP-1-fP
            mv_string2 = recv_strings_from_process(process, strings_to_recv)
            res_string2 = recv_strings_from_process(process, strings_to_recv)
            # decompress forward
            x_f_hat = model.decompress(x_ref,mv_string2,res_string2)
            # concate
            x_hat = torch.cat((torch.flip(x_b_hat,[0]),x_ref,x_f_hat),dim=0)
        else:
            strings_to_recv = GoP-1
            # receive two strings
            strings_to_recv = fP
            mv_string = recv_strings_from_process(process, strings_to_recv)
            res_string = recv_strings_from_process(process, strings_to_recv)
            # decompress backward
            x_f_hat = model.decompress(x_ref,mv_string,res_string)
            # concate
            x_hat = torch.cat((x_ref,x_f_hat),dim=0)
                
        if t_warmup is None:
            t_warmup = time.perf_counter() - t_0
            
        for com in x_hat:
            com = com.cuda().unsqueeze(0)
            raw = data[i].cuda().unsqueeze(0)
            psnr_list += [PSNR(raw, com)]
            i += 1
        # Count time
        total_time = time.perf_counter() - t_0
        fps = i/total_time
        # show result
        stream_iter.set_description(
            f"{i:3}. "
            f"FPS: {fps:.2f}. "
            f"PSNR: {float(psnr_list[-1]):.2f}. "
            f"Total: {total_time:.3f}. ")
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_list,fps,t_warmup
    
def RLVC_DVC_client(args,data,model=None,Q=None,fP=6,bP=6):
    block_until_open(args.server_ip,args.server_port)
    GoP = fP+bP+1
    # cannot connect before server is started
    # start a process to pipe data to netcat
    cmd = f'nc {args.server_ip} {args.server_port}'
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
    L = data.size(0)
    t_0 = None
    for i in range(L):
        # read data
        # wait for 1/30. or 1/60.
        while t_0 is not None and time.perf_counter() - t_0 < 1/60.:
            time.sleep(1/60.)
        t_0 = time.perf_counter()
        p = i%GoP
        if p > fP:
            # compress forward
            x_ref,mv_string,res_string,com_hidden,com_mv_prior_latent,com_res_prior_latent = \
                model.compress(x_ref, data[i:i+1], com_hidden, p>fP+1, com_mv_prior_latent, com_res_prior_latent)
            # send frame type
            bytes_send = struct.pack('B',2) # 0:iframe;1:backward;2 forward
            process.stdin.write(bytes_send)
            # send strings
            send_strings_to_process(process, [mv_string,res_string], useXZ=False)
            x_ref = x_ref.detach()
        elif p == fP or i == L-1:
            # get current GoP 
            x_GoP = data[i//GoP*GoP:i//GoP*GoP+GoP]
            x_b = torch.flip(x_GoP[:fP+1],[0])
            B,_,H,W = x_b.size()
            com_hidden = model.init_hidden(H,W)
            com_mv_prior_latent = com_res_prior_latent = None
            # send this compressed I frame
            x_ref = x_b[:1]
            # send frame type
            bytes_send = struct.pack('B',0) # 0:iframe;1:backward;2 forward
            process.stdin.write(bytes_send)
            # compress backward
            for j in range(1,B):
                x_ref,mv_string,res_string,com_hidden,com_mv_prior_latent,com_res_prior_latent = \
                    model.compress(x_ref, x_b[j:j+1], com_hidden, j>1, com_mv_prior_latent, com_res_prior_latent)
                x_ref = x_ref.detach()
                # send frame type
                bytes_send = struct.pack('B',1) # 0:iframe;1:backward;2 forward
                process.stdin.write(bytes_send)
                # send strings
                send_strings_to_process(process, [mv_string,res_string], useXZ=False)
            # init some states for forward compression
            com_hidden = model.init_hidden(H,W)
            com_mv_prior_latent = com_res_prior_latent = None
            x_ref = x_b[:1]
        else:
            # collect frames for current GoP
            pass
    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()

def RLVC_DVC_server(args,data,model=None,Q=None,fP=6,bP=6):
    GoP = fP+bP+1
    # Beginning time of streaming
    t_0 = time.perf_counter()
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.server_port}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    psnr_list = []
    t_warmup = None
    L = data.size(0)
    stream_iter = tqdm(range(L))
    for i in stream_iter:
        p = i%GoP
        # wait for 1/30. or 1/60.
        if p > fP:
            # [B=1] receive frame type
            bytes_recv = process.stdout.read(1)
            frame_type = struct.unpack('B',bytes_recv)[0]
            # receive strings
            mv_string = recv_strings_from_process(process, 1, useXZ=False)
            res_string = recv_strings_from_process(process, 1, useXZ=False)
            # decompress forward
            x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                model.decompress(x_ref, mv_string, res_string, decom_hidden, p>fP+1, decom_mv_prior_latent, decom_res_prior_latent)
            x_ref = x_ref.detach()
            psnr_list += [PSNR(data[i:i+1], x_ref)]
            # Count time
            total_time = time.perf_counter() - t_0
            fps = i/total_time
            # show result
            stream_iter.set_description(
                f"{i:3}. "
                f"FPS: {fps:.2f}. "
                f"PSNR: {float(psnr_list[-1]):.2f}. "
                f"Total: {total_time:.3f}. ")
        elif p == fP or i == L-1:
            # get current GoP 
            x_GoP = data[i//GoP*GoP:i//GoP*GoP+GoP]
            x_b = torch.flip(x_GoP[:fP+1],[0])
            B,_,H,W = x_b.size()
            decom_hidden = model.init_hidden(H,W)
            decom_mv_prior_latent = decom_res_prior_latent = None
            # [B=1] receive frame type
            bytes_recv = process.stdout.read(1)
            frame_type = struct.unpack('B',bytes_recv)[0]
            x_ref = x_b[:1]
            # get compressed I frame
            if t_warmup is None:
                t_warmup = time.perf_counter() - t_0
            # decompress backward
            psnr_list1 = []
            for j in range(1,B):
                # [B=1] receive frame type
                bytes_recv = process.stdout.read(1)
                frame_type = struct.unpack('B',bytes_recv)[0]
                # receive strings
                mv_string = recv_strings_from_process(process, 1, useXZ=False)
                res_string = recv_strings_from_process(process, 1, useXZ=False)
                x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                    model.decompress(x_ref, mv_string, res_string, decom_hidden, j>1, decom_mv_prior_latent, decom_res_prior_latent)
                x_ref = x_ref.detach()
                psnr_list1 += [PSNR(x_b[j:j+1], x_ref)]
                # Count time
                total_time = time.perf_counter() - t_0
                fps = i/total_time
                # show result
                stream_iter.set_description(
                    f"{i:3}. "
                    f"FPS: {fps:.2f}. "
                    f"PSNR: {float(psnr_list1[-1]):.2f}. "
                    f"Total: {total_time:.3f}. ")
            psnr_list += psnr_list1[::-1] + [torch.FloatTensor([40]).squeeze(0).to(data.device)]
            decom_hidden = model.init_hidden(H,W)
            decom_mv_prior_latent = decom_res_prior_latent = None
            x_ref = x_b[:1]
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_list,fps,t_warmup
        
def dynamic_simulation(args, test_dataset):
    # get server and client simulator
    if args.task in ['RLVC','DVC']:
        server_sim = RLVC_DVC_server
        client_sim = RLVC_DVC_client
    elif args.task in ['SPVC','AE3D']:
        server_sim = SPVC_AE3D_server
        client_sim = SPVC_AE3D_client
    elif args.task in ['x264','x265']:
        server_sim = x26x_server
        client_sim = x26x_client
    else:
        print('Unexpected task:',args.task)
        exit(1)
    ds_size = len(test_dataset)
    Q_list = [15,19,23,27] if args.Q_option == 'Slow' else [23]
    com_level_list = [0,1,2,3] if args.Q_option == 'Slow' else [2]
    for com_level,Q in zip(com_level_list,Q_list):
        ####### Load model
        if args.task in ['RLVC','DVC','SPVC','AE3D']:
            model = LoadModel(args.task,compression_level=com_level,use_cuda=args.use_cuda)
            model.eval()
        else:
            model = None
        ##################
        data = []
        psnr_module = AverageMeter()
        test_iter = tqdm(range(ds_size))
        for data_idx in test_iter:
            frame,eof = test_dataset[data_idx]
            if args.task in ['RLVC','DVC','SPVC','AE3D']:
                data.append(transforms.ToTensor()(frame))
            else:
                data.append(frame)
            if not eof:
                continue
            if args.task in ['RLVC','DVC','SPVC','AE3D']:
                data = torch.stack(data, dim=0)
                if args.use_cuda:
                    data = data.cuda()
            
            with torch.no_grad():
                if args.role == 'Standalone':
                    threading.Thread(target=client_sim, args=(args,data,model,Q)).start() 
                    psnr_list,fps,t_warmup = server_sim(args,data,model=model,Q=Q)
                elif args.role == 'Server':
                    psnr_list,fps,t_warmup = server_sim(args,data,model=model,Q=Q)
                elif args.role == 'Client':
                    client_sim(args,data,model=model,Q=Q)
                else:
                    print('Unexpected role:',args.role)
                    exit(1)
            
            # aggregate loss
            psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
            
            # record loss
            psnr_module.update(psnr.cpu().data.item())
            
            # show result
            if args.role == 'Standalone' or args.role == 'Server':
                test_iter.set_description(
                    f"L:{com_level}. "
                    f"{data_idx:6}. "
                    f"FPS: {fps:.2f}. "
                    f"Warmup: {t_warmup:.2f}. "
                    f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). ")
                
            # clear input
            data = []
            
        test_dataset.reset()
        if args.task in ['RLVC','DVC','SPVC','AE3D']:
            enc,dec = showTimer(model)
# todo: a protocol to send strings of compressed frames
# complete I frame comrpession
# run this script in docker
# then test throughput(fps) and rate-distortion on different devices and different losses
# need to add time measurement in parallel compression/decompress
# THROUGHPUT
# use locks 
# auto detect gpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--role', type=str, default='Standalone', help='Server or Client or Standalone')
    parser.add_argument('--dataset', type=str, default='UVG', help='UVG or MCL-JCV')
    parser.add_argument('--server_ip', type=str, default='localhost', help='Server IP')
    parser.add_argument('--server_port', type=str, default='8087', help='Server port')
    parser.add_argument('--probe_port', type=str, default='8086', help='Port to check if server is ready')
    parser.add_argument('--Q_option', type=str, default='Fast', help='Slow or Fast')
    parser.add_argument('--task', type=str, default='RLVC', help='RLVC,DVC,SPVC,AE3D,x265,x264')
    parser.add_argument('--mode', type=str, default='Dynamic', help='Dynamic or static simulation')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    
    # check gpu
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        args.use_cuda = False
        
    print(args)
    
    if args.dataset == 'UVG':
        test_dataset = VideoDataset('UVG', frame_size=(256,256))
    else:
        test_dataset = VideoDataset('../dataset/MCL-JCV', frame_size=(256,256))
        
    if args.mode == 'Dynamic':
        dynamic_simulation(args, test_dataset)
    else:
        if args.task in ['x264','x265']:
            static_simulation_x26x(args, test_dataset)
        elif args.task in ['RLVC','DVC','SPVC','AE3D']:
            static_simulation_model(args, test_dataset)