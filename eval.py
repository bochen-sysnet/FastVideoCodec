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

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def LoadModel(CODEC_NAME,compression_level = 2,use_split=False):
    loss_type = 'P'
    best_path = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_best.pth'
    ckpt_path = f'backup/{CODEC_NAME}/{CODEC_NAME}-{compression_level}{loss_type}_ckpt.pth'

    ####### Codec model 
    model = get_codec_model(CODEC_NAME,loss_type=loss_type,compression_level=compression_level,use_split=use_split)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('Total number of trainable codec parameters: {}'.format(pytorch_total_params))

    if model.name == 'DVC-pretrained':
        return model

    ####### Load codec model 
    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint['state_dict'])
        print("Loaded model codec score: ", checkpoint['score'])
        del checkpoint
    elif os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint['state_dict'])
        print("Loaded model codec score: ", checkpoint['score'])
        del checkpoint
    elif 'LSVC-L-128' in CODEC_NAME:
        psnr_list = [256,512,1024,2048]
        DVC_ckpt_name = f'DVC/snapshot/{psnr_list[compression_level]}.model'
        checkpoint = torch.load(DVC_ckpt_name,map_location=torch.device('cuda:0'))
        load_state_dict_all(model, checkpoint)
        print(f"Loaded model codec from {DVC_ckpt_name}")
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
    quality_levels = [7,11,15,19,23,27,31]
    
    Q_list = quality_levels[args.level_range[0]:args.level_range[1]] if args.Q_option == 'Slow' else [15]
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
                
            psnr_list,msssim_list,bpp_act_list,compt,decompt = compress_whole_video(args.task,data,Q,*test_dataset._frame_size)
            
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
    for lvl in range(args.level_range[0],args.level_range[1]):
        if args.Q_option != 'Slow' and lvl>0:continue
        model = LoadModel(args.task,compression_level=lvl,use_split=args.use_split)
        if args.use_cuda:
            if not args.use_split:
                if 'LSVC' in args.task:
                    model.parallel()
                else:
                    model = model.cuda()
        else:
            model = model.cpu()
        model.eval()
        aux_loss_module = AverageMeter()
        img_loss_module = AverageMeter()
        ba_loss_module = AverageMeter()
        be_loss_module = AverageMeter()
        psnr_module = AverageMeter()
        msssim_module = AverageMeter()
        all_loss_module = AverageMeter()
        compt_module = AverageMeter()
        decompt_module = AverageMeter()
        video_bpp_module = AverageMeter()
        ds_size = len(test_dataset)
        GoP = args.fP + args.bP +1
        GoP_meters = [AverageMeter() for _ in range(GoP)]
        GoP_meters2 = [AverageMeter() for _ in range(GoP)]
        WP_meters = [AverageMeter() for _ in range(GoP)]
        MC_meters = [AverageMeter() for _ in range(GoP)]
        data = []
        all_psnr_list = []
        test_iter = tqdm(range(ds_size))
        for data_idx,_ in enumerate(test_iter):
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
                if l>args.fP+1:
                    _,img_loss_list1,bpp_est_list1,aux_loss_list1,psnr_list1,msssim_list1,bpp_act_list1,encoding_time1,decoding_time1 = parallel_compression(model,torch.flip(data[:args.fP+1],[0]),True)
                    # data[args.fP:args.fP+1] = com_imgs[0:1]
                    _,img_loss_list2,bpp_est_list2,aux_loss_list2,psnr_list2,msssim_list2,bpp_act_list2,encoding_time2,decoding_time2 = parallel_compression(model,data[args.fP:],False)
                    if args.use_ep:
                        for idx,wp in enumerate(aux_loss_list1):
                            WP_meters[5-idx].update(wp)
                        for idx,mc in enumerate(msssim_list1):
                            MC_meters[5-idx].update(mc)
                        for idx,wp in enumerate(aux_loss_list2):
                            WP_meters[7+idx].update(wp)
                        for idx,mc in enumerate(msssim_list2):
                            MC_meters[7+idx].update(mc)
                    img_loss_list = img_loss_list1[::-1] + img_loss_list2
                    aux_loss_list = aux_loss_list1[::-1] + aux_loss_list2
                    psnr_list = psnr_list1[::-1] + psnr_list2
                    msssim_list = msssim_list1[::-1] + msssim_list2
                    bpp_act_list = bpp_act_list1[::-1] + bpp_act_list2
                    bpp_est_list = bpp_est_list1[::-1] + bpp_est_list2
                    l1,l2 = len(img_loss_list1),len(img_loss_list2)
                    encoding_time,decoding_time = (encoding_time1*l1+encoding_time2*l2)/(l1+l2),(decoding_time1*l1+decoding_time2*l2)/(l1+l2)
                else:
                    _,img_loss_list,bpp_est_list,aux_loss_list,psnr_list,msssim_list,bpp_act_list,encoding_time,decoding_time = parallel_compression(model,torch.flip(data,[0]),True)
                    
                # aggregate loss
                ba_loss = torch.stack(bpp_act_list,dim=0).mean(dim=0)
                psnr = torch.stack(psnr_list,dim=0).mean(dim=0)
                all_psnr_list += torch.stack(psnr_list,dim=0).tolist()
                
                # record loss
                ba_loss_module.update(ba_loss.cpu().data.item(), l)
                psnr_module.update(psnr.cpu().data.item(),l)

                compt_module.update(encoding_time,l)
                decompt_module.update(decoding_time,l)
                video_bpp_module.update(ba_loss,l)

                if aux_loss_list:
                    aux_loss = torch.stack(aux_loss_list,dim=0).mean(dim=0)
                    aux_loss_module.update(aux_loss.cpu().data.item(), l)
                    img_loss = torch.stack(img_loss_list,dim=0).mean(dim=0)
                    img_loss_module.update(img_loss.cpu().data.item(), l)
                    msssim = torch.stack(msssim_list,dim=0).mean(dim=0)
                    msssim_module.update(msssim.cpu().data.item(), l)

                # record psnr per position
                if args.use_ep:
                    for idx,p in enumerate(psnr_list):
                        GoP_meters[idx%GoP].update(p)
                    for idx,b in enumerate(bpp_act_list):
                        GoP_meters2[idx%GoP].update(b)
            
            # show result
            test_iter.set_description(
                f"{data_idx:6}. "
                f"IL: {img_loss_module.val:.2f} ({img_loss_module.avg:.2f}). "
                f"BA: {ba_loss_module.val:.4f} ({ba_loss_module.avg:.4f}). "
                f"P: {psnr_module.val:.2f} ({psnr_module.avg:.2f}). "
                f"M: {msssim_module.val:.4f} ({msssim_module.avg:.4f}). "
                f"A: {aux_loss_module.val:.4f} ({aux_loss_module.avg:.4f}). "
                f"I: {float(max(psnr_list)):.2f}. "
                f"E: {compt_module.avg:.3f} D: {decompt_module.avg:.3f}. ")
                
            # clear input
            data = []

            if eof:
                with open(f'{args.task}.log','a') as f:
                    f.write(f'{lvl},{video_bpp_module.avg:.4f},{compt_module.avg:.3f},{decompt_module.avg:.3f}\n')
                    f.write(str(all_psnr_list)+'\n')
                all_psnr_list = []
                compt_module.reset()
                decompt_module.reset()
                video_bpp_module.reset()

            
        test_dataset.reset()
        if args.use_ep:
            psnrs = [float(gm.avg) for gm in GoP_meters]
            bpps = [float(gm.avg) for gm in GoP_meters2]
            psnrs2 = [float(gm.avg) for gm in MC_meters]
            psnrs3 = [float(gm.avg) for gm in WP_meters]
            print(lvl,bpps)
            print(psnrs)
            print(psnrs2)
            print(psnrs3)
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
    # wait for server to test
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((args.client_ip, int(args.crdy_port)))
    s.listen(1)
    conn, addr = s.accept()
    #########################
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
    block_until_open(args.server_ip,args.srdy_port)
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
    # probe port
    conn.close()
    s.close()
    return fps,0
    
# how to direct rtsp traffic?
def x26x_server(args,data,model=None,Q=None,width=256,height=256):
    # only start if client is started
    block_until_open(args.client_ip,args.crdy_port)
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
    s.bind((args.server_ip, int(args.srdy_port)))
    s.listen(1)
    conn, addr = s.accept()
    psnr_module = AverageMeter()
    i = 0
    GoP = 1+args.fP+args.bP
    t_startup = None
    frame_count = 0
    t_rebuffer_total = r_rate = fps = 0
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
        if args.use_disp:
            resized = cv2.resize(frame, (512,512))
            cv2.imshow("H.265", resized)
            psnr_list = [35.79,35.26,33.46]
            bpp_list = [0.18,0.19,0.29]
            demoidx = int(args.dataset[-1])-1
            cv2.setWindowTitle("H.265", f"[H.265] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
            cv2.waitKey(1)

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
        # fps = i/total_time
        fps = frame_count/(total_time - t_startup) if t_startup is not None else 0
        r_rate = t_rebuffer_total/total_time
    
        # show result
        stream_iter.set_description(
            f"Frame count {i:3}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}) dB. "
            f"FPS: {fps:.2f}. "
            f"Rebuffering duration: {t_rebuffer_total:.2f} s. "
            f"Total duration: {total_time:.3f} s. ")
    conn.close()
    s.close()
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    cv2.destroyAllWindows()
    return fps,t_rebuffer_total/total_time,t_startup
            
def SPVC_AE3D_client(args,data,model=None,Q=None):
    # start a process to pipe data to netcat
    if not args.encoder_test:
        # wait for server to test
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((args.client_ip, int(args.crdy_port)))
        s.listen(1)
        conn, addr = s.accept()
        #########################
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
            if not args.encoder_test:
                while time.perf_counter() - t_0 < (i+args.fP+1)/args.fps:time.sleep(0.001)
            mv_string1,res_string1,_ = model.compress(x_b)
            # Send strings in order
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string1,res_string1])
            # compress forward
            x_f = x_GoP[args.fP:]
            # wait for forward
            if not args.encoder_test:
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
            if not args.encoder_test:
                while time.perf_counter() - t_0 < (i+GoP_size)/args.fps:time.sleep(0.001)
            # ready
            mv_string,res_string,bpp_act_list = model.compress(x_f)
            # Send strings in order
            if not args.encoder_test:
                send_strings_to_process(process, [mv_string,res_string])
        # Count time
        if t_first is not None:
            frame_count += GoP_size
            if args.encoder_test:
                frame_count -= 1
        else:
            t_first = time.perf_counter() - t_0
        total_time = time.perf_counter() - t_0
        fps = frame_count/(total_time-t_first)
        # progress bar
        encoder_iter.set_description(
            f"Encoder: {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Total: {total_time:.3f}. ")
    # GPU 
    gm = get_gpu_memory_map()
    if not args.encoder_test:
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()
        # probe port
        conn.close()
        s.close()
    return fps,gm[0]
    
def SPVC_AE3D_server(args,data,model=None,Q=None):
    # only start if client is started
    block_until_open(args.client_ip,args.crdy_port)
    t_0 = time.perf_counter()
    GoP = args.fP + args.bP +1
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.stream_port}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    # initialize
    t_startup = None # duration to receive the first frame
    frame_count = 0
    i = 0
    psnr_module = AverageMeter()
    t_rebuffer_total = 0
    r_rate = fps = 0
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
            n_mv = 1 if model.entropy_trick else bs
            n_res = 1 if model.entropy_trick else bs
            if bs<=2:
                n_res = 1
            elif bs<=6:
                n_res = 2
            else:
                n_res = 3
            mv_string1 = recv_strings_from_process(process, n_mv)
            res_string1 = recv_strings_from_process(process, n_res)
            # decompress backward
            x_b_hat = model.decompress(x_ref,mv_string1,res_string1,bs)
            # display
            if args.use_disp:
                for frame in torch.cat((torch.flip(x_b_hat,[0]),x_ref)):
                    frame = transforms.ToPILImage()(frame.squeeze(0))
                    frame = np.array(frame)
                    resized = cv2.resize(frame, (512,512))
                    cv2.imshow("LSVC", resized)
                    psnr_list = [32.65,36.06,33.31]
                    bpp_list = [0.27,0.22,0.39]
                    demoidx = int(args.dataset[-1])-1
                    cv2.setWindowTitle("LSVC", f"[LSVC] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
                    cv2.waitKey(1)
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
            n_mv = 1 if model.entropy_trick else bs
            n_res = 1 if model.entropy_trick else bs
            if bs<=2:
                n_res = 1
            elif bs<=6:
                n_res = 2
            else:
                n_res = 3
            mv_string2 = recv_strings_from_process(process, n_mv)
            res_string2 = recv_strings_from_process(process, n_res)
            # decompress forward
            x_f_hat = model.decompress(x_ref,mv_string2,res_string2,bs)
            # display
            if args.use_disp:
                for frame in x_f_hat:
                    frame = transforms.ToPILImage()(frame.squeeze(0))
                    frame = np.array(frame)
                    resized = cv2.resize(frame, (512,512))
                    cv2.imshow("LSVC", resized)
                    psnr_list = [32.65,36.06,33.31]
                    bpp_list = [0.27,0.22,0.39]
                    demoidx = int(args.dataset[-1])-1
                    cv2.setWindowTitle("LSVC", f"[LSVC] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
                    cv2.waitKey(1)
            # concate
            x_hat = torch.cat((torch.flip(x_b_hat,[0]),x_ref,x_f_hat),dim=0)
        else:
            bs = GoP_size-1
            # receive two strings
            n_mv = 1 if model.entropy_trick else bs
            n_res = 1 if model.entropy_trick else bs
            if bs<=2:
                n_res = 1
            elif bs<=6:
                n_res = 2
            else:
                n_res = 3
            mv_string = recv_strings_from_process(process, n_mv)
            res_string = recv_strings_from_process(process, n_res)
            # decompress backward
            x_f_hat = model.decompress(x_ref,mv_string,res_string,bs)
            # concate
            x_hat = torch.cat((x_ref,x_f_hat),dim=0)
            # display
            if args.use_disp:
                for frame in x_hat:
                    frame = transforms.ToPILImage()(frame.squeeze(0))
                    frame = np.array(frame)
                    resized = cv2.resize(frame, (512,512))
                    cv2.imshow("LSVC", resized)
                    psnr_list = [32.65,36.06,33.31]
                    bpp_list = [0.27,0.22,0.39]
                    demoidx = int(args.dataset[-1])-1
                    cv2.setWindowTitle("LSVC", f"[LSVC] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
                    cv2.waitKey(33)

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

        # i += GoP_size
        # fps = i/total_time
        total_time = time.perf_counter() - t_0
        fps = frame_count/(total_time - t_startup)
        r_rate = t_rebuffer_total/total_time

        # measure metrics
        for com in x_hat:
            com = com.unsqueeze(0)
            raw = data[i].unsqueeze(0)
            psnr = PSNR(com, raw).cpu().data.item()
            if psnr < 100:
                psnr_module.update(psnr)
            i += 1

        # show result
        stream_iter.set_description(
            f"Frame count: {i:3}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}) dB. "
            f"FPS: {fps:.2f}. "
            f"Rebuffering duration: {t_rebuffer_total:.2f} s. "
            f"Total duration: {total_time:.3f} s. ")
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    cv2.destroyAllWindows()
    return fps,t_rebuffer_total/total_time,t_startup
    
def RLVC_DVC_client(args,data,model=None,Q=None):
    if not args.encoder_test:
        # wait for server to test
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((args.client_ip, int(args.crdy_port)))
        s.listen(1)
        conn, addr = s.accept()
        #########################
        # start a process to pipe data to netcat
        block_until_open(args.server_ip,args.stream_port)
        cmd = f'nc {args.server_ip} {args.stream_port}'
        process = sp.Popen(shlex.split(cmd), stdin=sp.PIPE)
    GoP = args.fP + args.bP +1
    L = data.size(0)
    t_0 = time.perf_counter()
    encoder_iter = tqdm(range(L))
    frame_count = 0
    for i in encoder_iter:
        # wait for 1/30. or 1/60.
        if not args.encoder_test:
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
            com_hidden = model.init_hidden(H,W,x_b.device)
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
            com_hidden = model.init_hidden(H,W,x_b.device)
            com_mv_prior_latent = com_res_prior_latent = None
            x_ref = x_b[:1]

        # Count time
        total_time = time.perf_counter() - t_0
        if i%GoP!=0:frame_count += 1
        fps = frame_count/(total_time)

        # progress bar
        encoder_iter.set_description(
            f"Encoder: {i:3}. "
            f"FPS: {fps:.2f}. "
            f"Total: {total_time:.3f}. ")

    # GPU 
    gpu = get_gpu_memory_map()

    if not args.encoder_test:
        # Close and flush stdin
        process.stdin.close()
        # Wait for sub-process to finish
        process.wait()
        # Terminate the sub-process
        process.terminate()
        # probe port
        conn.close()
        s.close()
    return fps,gpu[0]

def RLVC_DVC_server(args,data,model=None,Q=None):
    # Beginning time of streaming
    # only start if client is started
    block_until_open(args.client_ip,args.crdy_port)
    t_0 = time.perf_counter()
    GoP = args.fP+args.bP+1
    # create a pipe for listening from netcat
    cmd = f'nc -lkp {args.stream_port}'
    process = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
    psnr_module = AverageMeter()
    L = data.size(0)
    t_rebuffer_total = fps = r_rate = 0
    t_startup = None
    frame_count = 0
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
                frame_count += 1
            x_ref = x_ref.detach()
            psnr_module.update(PSNR(data[i:i+1], x_ref).cpu().data.item())
            # display
            if args.use_disp:
                frame = transforms.ToPILImage()(x_ref.squeeze(0))
                frame = np.array(frame)
                resized = cv2.resize(frame, (512,512))
                cv2.imshow("RLVC", resized)
                psnr_list = [32.07,35.12,32.46]
                bpp_list = [0.18,0.14,0.24]
                demoidx = int(args.dataset[-1])-1
                cv2.setWindowTitle("RLVC", f"[RLVC] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
                cv2.waitKey(1)
        elif p == args.fP or i == L-1:
            # get current GoP 
            x_GoP = data[i//GoP*GoP:i//GoP*GoP+GoP]
            x_b = torch.flip(x_GoP[:args.fP+1],[0])
            B,_,H,W = x_b.size()
            decom_hidden = model.init_hidden(H,W,x_b.device)
            decom_mv_prior_latent = decom_res_prior_latent = None
            x_ref = x_b[:1]
            frame_list = [x_ref]
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
                frame_list = [x_ref] + frame_list
            decom_hidden = model.init_hidden(H,W,x_b.device)
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
                frame_count += B
            if args.use_disp:
                for frame in frame_list:
                    frame = transforms.ToPILImage()(frame.squeeze(0))
                    frame = np.array(frame)
                    resized = cv2.resize(frame, (512,512))
                    cv2.imshow("RLVC", resized)
                    psnr_list = [32.07,35.12,32.46]
                    bpp_list = [0.18,0.14,0.24]
                    demoidx = int(args.dataset[-1])-1
                    cv2.setWindowTitle("RLVC", f"[RLVC] {psnr_list[demoidx]:.2f}dB. {bpp_list[demoidx]:.2f}bpp. {fps:.2f}fps. {r_rate:.2f}. ")
                    cv2.waitKey(33)


        # Count time
        total_time = time.perf_counter() - t_0
        fps = frame_count/(total_time - t_startup) if t_startup is not None else 0
        r_rate = t_rebuffer_total/total_time
        # show result
        stream_iter.set_description(
            f"Frame count: {i:3}. "
            f"PSNR: {psnr_module.val:.2f} ({psnr_module.avg:.2f}) dB. "
            f"FPS: {fps:.2f}. "
            f"Rebuffering duration: {t_rebuffer_total:.2f} s. "
            f"Total duration: {total_time:.3f} s. ")
    # Close and flush stdin
    process.stdout.close()
    # Terminate the sub-process
    process.terminate()
    return fps,t_rebuffer_total/total_time,t_startup
        
def dynamic_simulation(args, test_dataset):
    # get server and client simulator
    if args.task in ['RLVC','DVC']:
        server_sim = RLVC_DVC_server
        client_sim = RLVC_DVC_client
    elif args.task in ['AE3D',] or 'SPVC' in args.task:
        server_sim = SPVC_AE3D_server
        client_sim = SPVC_AE3D_client
    elif args.task in ['x264','x265']:
        server_sim = x26x_server
        client_sim = x26x_client
    else:
        print('Unexpected task:',args.task)
        exit(1)   
    ds_size = len(test_dataset)
    Q_list = [15,19,23,27] if args.Q_option == 'Slow' else [15]
    com_level_list = [0,1,2,3] if args.Q_option == 'Slow' else [3]
    for com_level,Q in zip(com_level_list,Q_list):
        ####### Load model
        if args.task in ['RLVC','DVC','AE3D'] or 'SPVC' in args.task:
            model = LoadModel(args.task,compression_level=com_level,use_split=args.use_split)
            if args.use_cuda:
                model = model.cuda()
            else:
                model = model.cpu()
            model.eval()
        else:
            model = None
        ##################
        data = []
        latency_module = AverageMeter()
        fps_module = AverageMeter()
        rbr_module = AverageMeter()
        gpu_module = AverageMeter()
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
                if args.use_cuda:
                    data = data.cuda()
            
            with torch.no_grad():
                if args.role == 'standalone':
                    threading.Thread(target=client_sim, args=(args,data,model,Q)).start() 
                    fps,rebuffer_rate,latency = server_sim(args,data,model=model,Q=Q)
                elif args.role == 'server':
                    fps,rebuffer_rate,latency = server_sim(args,data,model=model,Q=Q)
                elif args.encoder_test or args.role == 'client':
                    fps,gpu = client_sim(args,data,model=model,Q=Q)
                    gpu_module.update(gpu)
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

        # write results
        with open(args.role + '.log','a+') as f:
            time_str = datetime.now().strftime("%d-%b-%Y(%H:%M:%S.%f)")
            outstr = f'{args.task} {args.fps} {com_level} ' +\
                    f'{fps_module.avg:.2f} {rbr_module.avg:.2f} '+\
                    f'{latency_module.avg:.2f} {gpu_module.avg:.2f}\n'
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
# x265    0.18,35.79  0.19,35.26  0.29,33.46
# LSVC-A  0.27,32.65  0.22,36.06  0.39,33.31
# RLVC    0.18,32.07  0.14,35.12  0.24,32.46


# fix gamma_p, change V and max buffer size (0-70s)
# bola-basic
# γ corresponds to how strongly we want to avoid rebuffering
# V buffer-perf metrics trade-off
# Q: buffer size
# set γp = 5 and varied V for different buffer sizes.
def BOLA_simulation():
    # how to derive bola parameters from S1,S2,v1,v2,v_M,Q_low,Q_max
    # alpha = (S1 * v2 - S2 * v1)/(S2 - S1)
    # V = (Q_max - Q_low) / (v_M - alpha)
    # gamma_p = (v_M * Q_low - alpha * Q_max) / (Q_max - Q_low)

    # V = .93 
    # gamma_p = 5
    # # S_m: bits per segment, p: second per segment
    # # v_m: utility or PSNR,υ_m = ln(S_m/S_1)
    # p = 1
    # T_k = 1
    # v_m = 35.0
    # Q = 0
    # S_m = 1
    # rho = (V * v_m + V * gamma_p - Q)/S_m
    # Q_next = max(Q-T_k/p) + 1

    with open(f'{args.task}.log','r') as f:
        for l in f.readlines():
            print(len(l))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--role', type=str, default='standalone', help='server or client or standalone')
    parser.add_argument('--dataset', type=str, default='UVG', help='UVG or MCL-JCV')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1', help='server IP')
    parser.add_argument('--client_ip', type=str, default='127.0.0.1', help='server IP')
    parser.add_argument('--stream_port', type=str, default='8846', help='RTSP port')
    parser.add_argument('--srdy_port', type=str, default='8847', help='Port to check if server is ready')
    parser.add_argument('--crdy_port', type=str, default='8848', help='Port to check if client is ready')
    parser.add_argument('--Q_option', type=str, default='Fast', help='Slow or Fast')
    parser.add_argument('--task', type=str, default='x264-veryfast', help='RLVC,DVC,SPVC,AE3D,x265,x264')
    parser.add_argument('--mode', type=str, default='static', help='dynamic or static simulation')
    parser.add_argument('--use_split', dest='use_split', action='store_true')
    parser.add_argument('--no-use_split', dest='use_split', action='store_false')
    parser.set_defaults(use_split=False)
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    parser.add_argument('--use_ep', dest='use error prop', action='store_true')
    parser.add_argument('--no-use_ep', dest='use error prop', action='store_false')
    parser.set_defaults(use_ep=False)
    parser.add_argument('--use_disp', dest='use_disp', action='store_true')
    parser.add_argument('--no-use_disp', dest='use_disp', action='store_false')
    parser.set_defaults(use_disp=False)
    parser.add_argument("--fP", type=int, default=6, help="The number of forward P frames")
    parser.add_argument("--bP", type=int, default=6, help="The number of backward P frames")
    parser.add_argument('--encoder_test', dest='encoder_test', action='store_true')
    parser.add_argument('--no-encoder_test', dest='encoder_test', action='store_false')
    parser.set_defaults(encoder_test=False)
    parser.add_argument("--channels", type=int, default=128, help="Channels of SPVC")
    parser.add_argument('--fps', type=float, default=1000., help='frame rate of sender')
    parser.add_argument('--target_rate', type=float, default=30., help='Target rate of receiver')
    parser.add_argument("--width", type=int, default=960, help="Input width")
    parser.add_argument("--height", type=int, default=640, help="Input height")
    parser.add_argument('--level_range', type=int, nargs='+', default=[0,7])
    args = parser.parse_args()
    
    # check gpu
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        args.use_split = False

    # check if test encoder
    if args.encoder_test:
        args.role = 'client'

    # setup streaming parameters
    if args.mode in['static','dynamic']:
        test_dataset = VideoDataset('../dataset/'+args.dataset, frame_size=(args.width,args.height))
        
    if args.mode == 'dynamic':
        assert(args.task in ['RLVC','DVC','x264','x265'] or 'SPVC' in args.task)
        dynamic_simulation(args, test_dataset)
    elif args.mode == 'static':
        assert(args.task in ['RLVC2','DVC-pretrained'] or 'LSVC' in args.task or 'x26' in args.task)
        if 'x26' in args.task:
            static_simulation_x26x(args, test_dataset)
        elif args.task in ['RLVC2','SPVC','DVC-pretrained'] or 'LSVC' in args.task:
            static_simulation_model(args, test_dataset)
    elif args.mode == 'bola':
        BOLA_simulation()
    else:
        print('Unknown mode:',args.mode)
