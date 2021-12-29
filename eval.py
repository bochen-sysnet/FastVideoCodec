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

from models import get_codec_model,parallel_compression,update_training,compress_whole_video,showTimer
from models import load_state_dict_whatever, load_state_dict_all, load_state_dict_only
from models import PSNR,MSSSIM

def LoadModel(CODEC_NAME,compression_level = 2,use_split=True):
    loss_type = 'P'
    RESUME_CODEC_PATH = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth'

    ####### Codec model 
    model = get_codec_model(CODEC_NAME,loss_type=loss_type,compression_level=compression_level,use_split=use_split)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    if model.name == 'DVC-pretrained':
        return model

    ####### Load codec model 
    if os.path.isfile(RESUME_CODEC_PATH):
        print("Loading for ", CODEC_NAME, 'from',RESUME_CODEC_PATH)
        checkpoint = torch.load(RESUME_CODEC_PATH,map_location=torch.device('cuda:0'))
        best_codec_score = checkpoint['score'][1:4]
        load_state_dict_all(model, checkpoint['state_dict'])
        print("Loaded model codec score: ", checkpoint['score'])
        del checkpoint
    else:
        print("Cannot load model codec", CODEC_NAME)
        #exit(1)
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
    for lvl in range(4):
        model = LoadModel(args.task,compression_level=lvl)
        model.eval()
        aux_loss_module = AverageMeter()
        img_loss_module = AverageMeter()
        ba_loss_module = AverageMeter()
        be_loss_module = AverageMeter()
        psnr_module = AverageMeter()
        msssim_module = AverageMeter()
        all_loss_module = AverageMeter()
        ds_size = len(test_dataset)
        GoP = args.fP + args.bP +1
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
                if l>args.fP+1:
                    com_imgs,img_loss_list1,bpp_est_list1,aux_loss_list1,psnr_list1,msssim_list1,bpp_act_list1 = parallel_compression(model,torch.flip(data[:args.fP+1],[0]),True)
                    data[args.fP:args.fP+1] = com_imgs[0:1]
                    _,img_loss_list2,bpp_est_list2,aux_loss_list2,psnr_list2,msssim_list2,bpp_act_list2 = parallel_compression(model,data[args.fP:],False)
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
    
def x26x_client(args,data,model=None,Q=None,width=256,height=256):
    fps = 25
    GOP = args.fP + args.bP +1
    if args.task == 'x265':
        cmd = f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p '+\
                f'-preset veryfast -tune zerolatency -x265-params "crf={Q}:keyint={GOP}:verbose=1" '+\
                f'-rtsp_transport tcp -f rtsp rtsp://{args.server_ip}:{args.stream_port}/live'
    elif args.task == 'x264':
        cmd = f'/usr/bin/ffmpeg -hide_banner -loglevel error -y -s {width}x{height} -pixel_format bgr24 -f rawvideo -r {fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p '+\
                f'-preset veryfast -tune zerolatency -crf {Q} -g {GOP} -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug '+\
                f'-rtsp_transport tcp -f rtsp rtsp://{args.server_ip}:{args.stream_port}/live'
    else:
        print('Codec not supported')
        exit(1)
    block_until_open(args.server_ip,args.probe_port)
    # create a rtsp pipe
    process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    t_0 = time.perf_counter()
    encoder_iter = tqdm(range(len(data)))
    for i in encoder_iter:
        # read data
        # wait for 1/30. or 1/60.
        img = np.array(data[i])
        while time.perf_counter() < t_0 + i/args.fps:time.sleep(0.001)
        # send image
        process.stdin.write(img.tobytes())
        # Count time
        total_time = time.perf_counter() - t_0
        fps = (i+1)/total_time
        # progress bar
        encoder_iter.set_description(
            f"Encoder: {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Total: {total_time:.3f}. ")
    # Close and flush stdin
    process.stdin.close()
    # Wait for sub-process to finish
    process.wait()
    # Terminate the sub-process
    process.terminate()
    return fps
    
# how to direct rtsp traffic?
def x26x_server(args,data,model=None,Q=None,width=256,height=256):
    # Beginning time of streaming
    t_0 = time.perf_counter()
    # create a rtsp server or listener
    # ssh -R [REMOTE:]REMOTE_PORT:DESTINATION:DESTINATION_PORT [USER@]SSH_SERVER
    # ssh -R 8555:localhost:8555 uiuc@192.168.251.195
    command = ['/usr/bin/ffmpeg',
        '-hide_banner', '-loglevel', 'error',
        '-rtsp_flags', 'listen',
        '-i', f'rtsp://{args.server_ip}:{args.stream_port}/live?tcp?',
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
    psnr_module = AverageMeter()
    i = 0
    GoP = 1+args.fP+args.bP
    t_startup = None
    frame_count = 0
    t_rebuffer_total = 0
    stream_iter = tqdm(range(len(data)))
    while True:
        # read width*height*3 bytes from stdout (1 frame)
        raw_frame = process.stdout.read(width*height*3)

        if len(raw_frame) != (width*height*3):
            #print('Error reading frame!!!')  # Break the loop in case of an error (too few bytes were read).
            break

        # Convert the bytes read into a NumPy array, and reshape it to video frame dimensions
        frame = np.fromstring(raw_frame, np.uint8)
        frame = frame.reshape((height, width, 3))

        if t_startup is None:
            if i==GoP:
                t_startup = time.perf_counter()-t_0
                t_replay = time.perf_counter()
                t_cache = GoP/args.target_rate
        else:
            t_ready = time.perf_counter()
            if t_ready > t_replay + 1/args.target_rate:
                # remove frame from cache
                t_cache -= t_ready - (t_replay + 1/args.target_rate)
            t_replay = t_replay + 1/args.target_rate # time when this frame finishes
            if t_cache < 0:
                t_rebuffer = -t_cache
                t_replay += t_rebuffer
                t_rebuffer_total += t_rebuffer
                t_cache = 0
            frame_count += 1
        
        # process metrics
        com = transforms.ToTensor()(frame).cuda().unsqueeze(0)
        raw = data[i+1] if args.task == 'x264' else data[i]
        raw = transforms.ToTensor()(raw).cuda().unsqueeze(0)
        psnr_module.update(PSNR(com, raw).cpu().data.item())
        i += 1
        
        # Count time
        total_time = time.perf_counter() - t_0
        fps = i/total_time
        # fps = frame_count/(total_time - t_startup) if t_startup is not None else 0
    
        # show result
        stream_iter.set_description(
            f"Decoder {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Rebuffer: {t_rebuffer_total:.2f}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"Total: {total_time:.3f}. ")
    conn.close()
    s.close()
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_module.avg,fps,t_rebuffer_total/total_time,t_startup
            
def SPVC_AE3D_client(args,data,model=None,Q=None):
    # start a process to pipe data to netcat
    if not args.encoder_test:
        block_until_open(args.server_ip,args.stream_port)
        cmd = f'nc {args.server_ip} {args.stream_port}'
        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
    t_0 = time.perf_counter()
    t_first = None
    frame_count = 0
    GoP = args.fP + args.bP +1
    L = data.size(0)
    encoder_iter = tqdm(range(0,L,GoP))
    for i in encoder_iter:
        x_GoP = data[i:i+GoP]
        GoP_size = x_GoP.size(0)
        # Send compressed I frame (todo)
        if GoP_size>args.fP+1:
            # compress I
            # compress backward
            x_b = torch.flip(x_GoP[:args.fP+1],[0])
            # wait for I and backward
            while time.perf_counter() - t_0 < (i+args.fP+1)/args.fps:time.sleep(0.001)
            mv_string1,res_string1,_ = model.compress(x_b)
            # Send strings in order
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string1,res_string1])
            # compress forward
            x_f = x_GoP[args.fP:]
            # wait for forward
            while time.perf_counter() - t_0 < (i+GoP_size)/args.fps:time.sleep(0.001)
            # ready
            mv_string2,res_string2,_ = model.compress(x_f)
            # Send strings in order
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string2,res_string2])
        else:
            # compress I
            # compress backward
            x_f = x_GoP
            # wait for backward
            while time.perf_counter() - t_0 < (i+GoP_size)/args.fps:time.sleep(0.001)
            # ready
            mv_string,res_string,bpp_act_list = model.compress(x_f)
            # Send strings in order
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string,res_string])
        # Count time
        if t_first is not None:
            frame_count += GoP_size
        else:
            t_first = time.perf_counter() - t_0
        total_time = time.perf_counter() - t_0
        fps = frame_count/(total_time-t_first)
        # progress bar
        encoder_iter.set_description(
            f"Encoder: {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Total: {total_time:.3f}. ")
    if not args.encoder_test:
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()
    return fps
    
def SPVC_AE3D_server(args,data,model=None,Q=None):
    t_0 = time.perf_counter()
    GoP = args.fP + args.bP +1
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.stream_port}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    # initialize
    t_startup = None # duration to receive the first frame
    frame_count = 0
    psnr_module = AverageMeter()
    i = 0
    t_rebuffer_total = 0
    L = data.size(0)
    stream_iter = tqdm(range(0,L,GoP))
    # start listening
    for begin in stream_iter:
        # decompress I frame
        x_GoP = data[begin:begin+GoP]
        x_ref = x_GoP[args.fP:args.fP+1] if x_GoP.size(0)>args.fP+1 else x_GoP[:1]
        GoP_size = x_GoP.size(0)
        # receive strings based on gop size
        if GoP_size>args.fP+1:
            bs = args.fP
            # receive the first two strings
            n_str = 1 if model.entropy_trick else bs
            mv_string1 = recv_strings_from_process(process, n_str)
            res_string1 = recv_strings_from_process(process, n_str)
            # decompress backward
            x_b_hat = model.decompress(x_ref,mv_string1,res_string1,bs)
            # rebuffer
            if t_startup is not None:
                # count rebuffer time
                t_ready = time.perf_counter()
                if t_ready > t_replay + bs/args.target_rate:
                    # remove frame from cache
                    t_cache -= t_ready - (t_replay + bs/args.target_rate)
                t_replay = t_replay + bs/args.target_rate # time when this frame finishes
                if t_cache < 0:
                    t_rebuffer = -t_cache
                    t_replay += t_rebuffer
                    t_rebuffer_total += t_rebuffer
                    t_cache = 0
            # current batch
            bs = GoP_size-1-args.fP
            # receive the second two strings
            n_str = 1 if model.entropy_trick else bs
            mv_string2 = recv_strings_from_process(process, n_str)
            res_string2 = recv_strings_from_process(process, n_str)
            # decompress forward
            x_f_hat = model.decompress(x_ref,mv_string2,res_string2,bs)
            # concate
            x_hat = torch.cat((torch.flip(x_b_hat,[0]),x_ref,x_f_hat),dim=0)
        else:
            bs = GoP_size-1
            # receive two strings
            n_str = 1 if model.entropy_trick else bs
            mv_string = recv_strings_from_process(process, n_str)
            res_string = recv_strings_from_process(process, n_str)
            # decompress backward
            x_f_hat = model.decompress(x_ref,mv_string,res_string,bs)
            # concate
            x_hat = torch.cat((x_ref,x_f_hat),dim=0)

        # start rebuffering after receiving a gop
        if t_startup is None:
            t_startup = time.perf_counter() - t_0
            t_replay = time.perf_counter()
            t_cache = GoP/args.target_rate
        else:
            # count rebuffer time
            t_ready = time.perf_counter()
            if t_ready > t_replay + bs/args.target_rate:
                # remove frame from cache
                t_cache -= t_ready - (t_replay + bs/args.target_rate)
            t_replay = t_replay + (bs+1)/args.target_rate # time when this frame finishes
            if t_cache < 0:
                t_rebuffer = -t_cache
                t_replay += t_rebuffer
                t_rebuffer_total += t_rebuffer
                t_cache = 0
        frame_count += GoP_size

        total_time = time.perf_counter() - t_0
        fps = frame_count/total_time

        # measure metrics
        for com in x_hat:
            com = com.unsqueeze(0)
            raw = data[i].unsqueeze(0)
            psnr_module.update(PSNR(com, raw).cpu().data.item())
            i += 1

        # show result
        stream_iter.set_description(
            f"Decoder: {i:3}. "
            f"Rebuffer: {t_rebuffer_total:.2f}. "
            f"FPS: {fps:.2f}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"Total: {total_time:.3f}. ")
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_module.avg,fps,t_rebuffer_total/total_time,t_startup
    
def RLVC_DVC_client(args,data,model=None,Q=None):
    if not args.encoder_test:
        # cannot connect before server is started
        # start a process to pipe data to netcat
        block_until_open(args.server_ip,args.stream_port)
        cmd = f'nc {args.server_ip} {args.stream_port}'
        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
    GoP = args.fP + args.bP +1
    L = data.size(0)
    t_0 = time.perf_counter()
    encoder_iter = tqdm(range(L))
    for i in encoder_iter:
        # wait for 1/30. or 1/60.
        while time.perf_counter() < t_0 + i/args.fps:time.sleep(0.001)
        p = i%GoP
        if p > args.fP:
            # compress forward
            x_ref,mv_string,res_string,com_hidden,com_mv_prior_latent,com_res_prior_latent = \
                model.compress(x_ref, data[i:i+1], com_hidden, p>args.fP+1, com_mv_prior_latent, com_res_prior_latent)
            # send strings
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string,res_string], useXZ=False)
            x_ref = x_ref.detach()
        elif p == args.fP or i == L-1:
            # get current GoP 
            x_GoP = data[i//GoP*GoP:i//GoP*GoP+GoP]
            x_b = torch.flip(x_GoP[:args.fP+1],[0])
            B,_,H,W = x_b.size()
            com_hidden = model.init_hidden(H,W)
            com_mv_prior_latent = com_res_prior_latent = None
            # send this compressed I frame
            x_ref = x_b[:1]
            # compress backward
            for j in range(1,B):
                # compress
                x_ref,mv_string,res_string,com_hidden,com_mv_prior_latent,com_res_prior_latent = \
                    model.compress(x_ref, x_b[j:j+1], com_hidden, j>1, com_mv_prior_latent, com_res_prior_latent)
                x_ref = x_ref.detach()
                # send strings
                if not args.encoder_test:
                    send_strings_to_process(process, [mv_string,res_string], useXZ=False)
            # init some states for forward compression
            com_hidden = model.init_hidden(H,W)
            com_mv_prior_latent = com_res_prior_latent = None
            x_ref = x_b[:1]

        # Count time
        total_time = time.perf_counter() - t_0
        fps = (i)/(total_time)
        # progress bar
        encoder_iter.set_description(
            f"Encoder: {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Total: {total_time:.3f}. ")
    if not args.encoder_test:
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()
    return fps

def RLVC_DVC_server(args,data,model=None,Q=None):
    # Beginning time of streaming
    t_0 = time.perf_counter()
    GoP = args.fP+args.bP+1
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.stream_port}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    psnr_module = AverageMeter()
    L = data.size(0)
    t_rebuffer_total = 0
    t_startup = None
    stream_iter = tqdm(range(L))
    for i in stream_iter:
        p = i%GoP
        # wait for 1/30. or 1/60.
        if p > args.fP:
            # receive strings
            mv_string = recv_strings_from_process(process, 1, useXZ=False)
            res_string = recv_strings_from_process(process, 1, useXZ=False)
            # decompress forward
            x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                model.decompress(x_ref, mv_string, res_string, decom_hidden, p>args.fP+1, decom_mv_prior_latent, decom_res_prior_latent)
            # replay
            if t_startup is None:
                if (p==GoP-1 or i==L-1):
                    t_startup = time.perf_counter()-t_0
                    t_replay = time.perf_counter()
                    t_cache = GoP/args.target_rate
            else:
                t_ready = time.perf_counter()
                if t_ready > t_replay + 1/args.target_rate:
                    # remove frame from cache
                    t_cache -= t_ready - (t_replay + 1/args.target_rate)
                t_replay = t_replay + 1/args.target_rate # time when this frame finishes
                if t_cache < 0:
                    t_rebuffer = -t_cache
                    t_replay += t_rebuffer
                    t_rebuffer_total += t_rebuffer
                    t_cache = 0
            x_ref = x_ref.detach()
            psnr_module.update(PSNR(data[i:i+1], x_ref).cpu().data.item())
        elif p == args.fP or i == L-1:
            # get current GoP 
            x_GoP = data[i//GoP*GoP:i//GoP*GoP+GoP]
            x_b = torch.flip(x_GoP[:args.fP+1],[0])
            B,_,H,W = x_b.size()
            decom_hidden = model.init_hidden(H,W)
            decom_mv_prior_latent = decom_res_prior_latent = None
            x_ref = x_b[:1]
            # get compressed I frame
            # decompress backward
            for j in range(1,B):
                # receive strings
                mv_string = recv_strings_from_process(process, 1, useXZ=False)
                res_string = recv_strings_from_process(process, 1, useXZ=False)
                x_ref,decom_hidden,decom_mv_prior_latent,decom_res_prior_latent = \
                    model.decompress(x_ref, mv_string, res_string, decom_hidden, j>1, decom_mv_prior_latent, decom_res_prior_latent)
                # record time
                x_ref = x_ref.detach()
                psnr_module.update(PSNR(x_b[j:j+1], x_ref).cpu().data.item())
            decom_hidden = model.init_hidden(H,W)
            decom_mv_prior_latent = decom_res_prior_latent = None
            x_ref = x_b[:1]
            if t_startup is not None:
                t_ready = time.perf_counter()
                if t_ready > t_replay + args.fP/args.target_rate:
                    # remove frame from cache
                    t_cache -= t_ready - (t_replay + args.fP/args.target_rate)
                t_replay = t_replay + (args.fP+1)/args.target_rate # time when this frame finishes
                if t_cache < 0:
                    t_rebuffer = -t_cache
                    t_replay += t_rebuffer
                    t_rebuffer_total += t_rebuffer
                    t_cache = 0
        # Count time
        total_time = time.perf_counter() - t_0
        fps = (i+1)/total_time
        # show result
        stream_iter.set_description(
            f"Decoder: {i:3}. "
            f"Rebuffer: {t_rebuffer_total:.2f}. "
            f"FPS: {fps:.2f}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
            f"Total: {total_time:.3f}. ")
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return psnr_module.avg,fps,t_rebuffer_total/total_time,t_startup
        
def dynamic_simulation(args, test_dataset):
    # get server and client simulator
    if args.task in ['RLVC','DVC']:
        server_sim = RLVC_DVC_server
        client_sim = RLVC_DVC_client
    elif args.task in ['AE3D'] or 'SPVC' in args.task:
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
        if args.task in ['RLVC','DVC','AE3D'] or 'SPVC' in args.task:
            model = LoadModel(args.task,compression_level=com_level,use_split=args.use_split)
            model.eval()
        else:
            model = None
        ##################
        data = []
        latency_module = AverageMeter()
        fps_module = AverageMeter()
        rbr_module = AverageMeter()
        load_iter = tqdm(range(ds_size))
        for data_idx in load_iter:
            frame,eof = test_dataset[data_idx]
            if args.task in ['RLVC','DVC','AE3D'] or 'SPVC' in args.task:
                data.append(transforms.ToTensor()(frame))
            else:
                data.append(frame)
            if not eof:
                continue
            if args.task in ['RLVC','DVC','AE3D'] or 'SPVC' in args.task:
                data = torch.stack(data, dim=0)
                data = data.cuda()
            
            with torch.no_grad():
                if args.role == 'standalone':
                    threading.Thread(target=client_sim, args=(args,data,model,Q)).start() 
                    psnr,fps,rebuffer_rate,latency = server_sim(args,data,model=model,Q=Q)
                elif args.role == 'server':
                    psnr,fps,rebuffer_rate,latency = server_sim(args,data,model=model,Q=Q)
                elif args.role == 'client' or args.encoder_test:
                    fps = client_sim(args,data,model=model,Q=Q)
                else:
                    print('Unexpected role:',args.role)
                    exit(1)
            
            # record loss
            fps_module.update(fps)
            if args.role == 'standalone' or args.role == 'server':
                rbr_module.update(rebuffer_rate)
                latency_module.update(latency)

            # clear input
            data = []

        # destroy model
        if 'SPVC' in args.task:
            model.destroy()

        # write results
        with open(args.role + '.log','a+') as f:
            time_str = datetime.now().strftime("%d-%b-%Y(%H:%M:%S.%f)")
            outstr = f'{time_str} {args.task} {com_level} ' +\
                    f'{fps_module.avg:.2f} {rbr_module.avg:.2f} {latency_module.avg:.2f}\n'
            f.write(outstr)
            if args.task in ['RLVC','DVC','AE3D'] or 'SPVC' in args.task:
                enc_str,dec_str,_,_ = showTimer(model)
                if args.role == 'standalone' or args.role == 'client':
                    f.write(enc_str+'\n')
                if args.role == 'standalone' or args.role == 'server':
                    f.write(dec_str+'\n')
            
        test_dataset.reset()

            
# maybe just 1080/2080 to 2070 and changing packet loss
# two server test
# delay: sudo tc qdisc add dev lo root netem delay 100ms 10ms
# loss: sudo tc qdisc add dev lo root netem loss 10%
# remove: sudo tc qdisc del dev lo root
# startup time, rebuffering, and video quality.
# Rebuffering ratio is the ratio between the rebuffering duration and the actual duration of video that played (rebuffering duration / playback duration)
# sudo ufw allow 53
# sudo ufw status verbose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--role', type=str, default='standalone', help='server or client or standalone')
    parser.add_argument('--dataset', type=str, default='UVG', help='UVG or MCL-JCV')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help='server IP')
    parser.add_argument('--stream_port', type=str, default='8846', help='RTSP port')
    parser.add_argument('--probe_port', type=str, default='8847', help='Port to check if server is ready')
    parser.add_argument('--Q_option', type=str, default='Fast', help='Slow or Fast')
    parser.add_argument('--task', type=str, default='RLVC', help='RLVC,DVC,SPVC,AE3D,x265,x264')
    parser.add_argument('--mode', type=str, default='dynamic', help='dynamic or static simulation')
    parser.add_argument('--use_split', dest='use_split', action='store_true')
    parser.add_argument('--no-use_split', dest='use_split', action='store_false')
    parser.set_defaults(use_split=False)
    parser.add_argument('--use_psnr', dest='use_psnr', action='store_true')
    parser.add_argument('--no-use_psnr', dest='use_psnr', action='store_false')
    parser.set_defaults(use_psnr=False)
    parser.add_argument("--fP", type=int, default=6, help="The number of forward P frames")
    parser.add_argument("--bP", type=int, default=6, help="The number of backward P frames")
    parser.add_argument('--encoder_test', dest='encoder_test', action='store_true')
    parser.add_argument('--no-encoder_test', dest='use_psnr', action='store_false')
    parser.set_defaults(encoder_test=False)
    parser.add_argument("--channels", type=int, default=128, help="Channels of SPVC")
    parser.add_argument('--fps', type=float, default=30., help='frame rate of sender')
    parser.add_argument('--target_rate', type=float, default=30., help='Target rate of receiver')
    args = parser.parse_args()
    
    # check gpu
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        args.use_split = False

    # check if test encoder
    if args.encoder_test:
        args.role = 'client'

    # setup streaming parameters

        
    print(args)
    assert args.dataset in ['UVG','MCL-JCV','Xiph','Xiph2']
    test_dataset = VideoDataset('../dataset/'+args.dataset, frame_size=(256,256))
        
    if args.mode == 'dynamic':
        dynamic_simulation(args, test_dataset)
    else:
        if args.task in ['x264','x265']:
            static_simulation_x26x(args, test_dataset)
        elif args.task in ['RLVC','DVC','SPVC96','SPVC','AE3D','DVC-pretrained']:
            static_simulation_model(args, test_dataset)