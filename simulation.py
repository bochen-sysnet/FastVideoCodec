import argparse
import re
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labelsize_b = 14
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#1C4670','#FF9636','#9D5FFB','#21B6A8','#D65780']
# colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
# colors =['#FD0707','#0D0DDF','#129114','#DDDB03','#FF8A12','#8402AD']
markers = ['o','^','s','>','P','D']
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
# linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]
    

GOP = 16
width,height = 2048,1024
pix_per_frame = width*height

def BOLA_simulation(total_traces = 100,
    tasks = ['LSVC-A','LSVC-L-128','RLVC2','x264-veryfast','x264-medium','x264-veryslow','x265-veryfast','x265-medium','x265-veryslow']):
    # read network traces
    import csv
    single_trace_len = 500
    downthrpt = []
    latency = []
    if args.trace_id == 0:
        trace_file = '../curr_videostream.csv'
        args.trace_dur = 10
    else:
        trace_file = '../curr_httpgetmt.csv'
        args.trace_dur = 5
    with open(trace_file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_cnt = 0
        for row in csv_reader:
            # if args.hardware == '1080':
            #     if line_cnt < single_trace_len * total_traces:
            #         line_cnt += 1
            #         continue
            # elif args.hardware == '3090':
            #     if line_cnt < single_trace_len * total_traces /10:
            #         line_cnt += 1
            #         continue
            # bytes per second to bps
            # micro seconds to sec
            if args.trace_id == 0:
                downthrpt_sample = float(row["downthrpt"])*8
                latency_sample = float(row["latency"])/1e6
            else:
                if row["bytes_sec_interval"] == 'NULL':
                    continue
                downthrpt_sample = float(row["bytes_sec_interval"])*8
                latency_sample = 0
            if downthrpt_sample>1e6 and downthrpt_sample<100e6:
                downthrpt += [downthrpt_sample]
                latency += [latency_sample]
            if len(downthrpt) >= single_trace_len * total_traces:
                break
    print('Trace stats:',np.array(downthrpt).mean(),np.array(downthrpt).std(),np.array(downthrpt).max(),np.array(downthrpt).min())

    QoE_matrix = []
    quality_matrix = []
    rebuffer_matrix = []
    stall_matrix = []
    all_mean_psnr,all_mean_bpp = [],[]
    all_dect_mean = [];all_dect_std = []
    for task in tasks:
        all_psnr,all_bitrate,all_dect,mean_psnr,mean_bpp = task_to_video_trace(task)
        all_mean_psnr += [mean_psnr]
        all_mean_bpp += [mean_bpp]
        all_dect_mean += [all_dect.mean()/GOP]
        all_dect_std += [all_dect.std()/GOP]
        QoE_list = [];quality_list = [];rebuffer_list = [];stall_list = []
        sim_iter = tqdm(range(total_traces))
        for _,i in enumerate(sim_iter):
            trace_start = i * single_trace_len
            trace_end = trace_start + single_trace_len
            QoE,quality,rebuffer,stall_freq = simulate_over_traces(all_psnr,all_bitrate,all_dect,downthrpt[trace_start:trace_end],latency[trace_start:trace_end],i)
            QoE_list += [QoE];quality_list += [quality];rebuffer_list += [rebuffer];stall_list += [stall_freq]
            sim_iter.set_description(
                f"{i:3}. "
                f"Task:{task}. "
                f"QoE:{QoE:.1f}."
                f"quality:{quality:.1f}. "
                f"rebuffer:{rebuffer:.4f}. "
                f"stall:{stall_freq:.3f}. ")
        QoE_matrix += [QoE_list];quality_matrix += [quality_list];rebuffer_matrix += [rebuffer_list];stall_matrix += [stall_list]
        sim_iter.reset()
    with open(f'/home/bo/Dropbox/Research/SIGCOMM23-VC/data/QoE_{args.trace_id}_{args.hardware}_{args.num_traces}.data','w') as f:
        f.write(str(QoE_matrix))
    with open(f'/home/bo/Dropbox/Research/SIGCOMM23-VC/data/quality_{args.trace_id}_{args.hardware}_{args.num_traces}.data','w') as f:
        f.write(str(quality_matrix))
    with open(f'/home/bo/Dropbox/Research/SIGCOMM23-VC/data/rebuffer_{args.trace_id}_{args.hardware}_{args.num_traces}.data','w') as f:
        f.write(str(rebuffer_matrix))
    with open(f'/home/bo/Dropbox/Research/SIGCOMM23-VC/data/stall_{args.trace_id}_{args.hardware}_{args.num_traces}.data','w') as f:
        f.write(str(stall_matrix))
    QoE_matrix = np.array(QoE_matrix);quality_matrix = np.array(quality_matrix);rebuffer_matrix = np.array(rebuffer_matrix);stall_matrix = np.array(stall_matrix)
    print('QOE:',QoE_matrix.mean(axis=1).tolist(),QoE_matrix.std(axis=1).tolist())
    print('Quality:',quality_matrix.mean(axis=1).tolist(),quality_matrix.std(axis=1).tolist())
    print('Rebuffer:',rebuffer_matrix.mean(axis=1).tolist(),rebuffer_matrix.std(axis=1).tolist())
    print('Stall:',stall_matrix.mean(axis=1).tolist(),stall_matrix.std(axis=1).tolist())
    # print(all_mean_psnr)
    # print(all_mean_bpp)
    # print(all_dect_mean,all_dect_std)

def task_to_video_trace(task):
    frame_psnr_dict = {}
    frame_bpp_dict = {}
    frame_dect_dict = {}
    with open(f'../{task}.log','r') as f:
        line_count = 0
        for l in f.readlines():
            if line_count%2 == 0:
                l = l.split(',')
                lvl,bpp,enct,dect = int(l[0]),float(l[1]),float(l[2]),float(l[3])
                # estimate the encoding and decoding and put it here
                if 'VC' in task:
                    # encdec1080=[[0.0324,0.0188];[0.0402,0.0285];[0.0632,0.0475]]
                    # encdec2080=[[0.0195,0.0093];[0.028,0.017];[0.0526,0.0408]]
                    if args.hardware == '2080':
                        dect_list = [0.0195,0.028,0.0526]
                    elif args.hardware == '1080':
                        dect_list = [0.0310,0.0382,0.0581]
                    elif args.hardware == '3090':
                        dect_list = [0.0102,0.01,0.012]
                    else:
                        print('Unknown hardware:',args.hardware)
                        exit(0)
                    if task == 'LSVC-A':
                        dect = dect_list[0]
                    elif task == 'LSVC-L-128':
                        dect = dect_list[1]
                    elif task == 'RLVC2':
                        dect = dect_list[2]
                    else:
                        print('Unknown trace:',task)
                        exti(0)
            else:
                if lvl not in frame_psnr_dict:
                    frame_psnr_dict[lvl] = []
                    frame_bpp_dict[lvl] = []
                    frame_dect_dict[lvl] = []
                l = l[1:-2].split(',')
                l = np.char.strip(l)
                psnr_list = np.array(l).astype(float).tolist()
                frame_psnr_dict[lvl] += psnr_list
                frame_bpp_dict[lvl] += [bpp] * len(psnr_list)
                frame_dect_dict[lvl] += [dect] * len(psnr_list)
            line_count += 1

    all_psnr = []
    all_bitrate = []
    all_dect = []
    pix_per_sec = pix_per_frame*args.fps
    num_segments = len(frame_psnr_dict[0])//GOP
    for lvl in range(len(frame_psnr_dict.keys())):
        frame_psnr_dict[lvl] = np.array(frame_psnr_dict[lvl][:GOP*num_segments]).reshape(num_segments,GOP).mean(axis=-1)
        all_psnr += [frame_psnr_dict[lvl].tolist()]
        del frame_psnr_dict[lvl]
        frame_bpp_dict[lvl] = np.array(frame_bpp_dict[lvl][:GOP*num_segments]).reshape(num_segments,GOP).mean(axis=-1) * pix_per_sec
        all_bitrate += [frame_bpp_dict[lvl].tolist()]
        del frame_bpp_dict[lvl]
        frame_dect_dict[lvl] = np.array(frame_dect_dict[lvl][:GOP*num_segments]).reshape(num_segments,GOP).mean(axis=-1) * GOP
        all_dect += [frame_dect_dict[lvl].tolist()]
        del frame_dect_dict[lvl]
    all_psnr = np.array(all_psnr)
    all_bitrate = np.array(all_bitrate)
    all_dect = np.array(all_dect)
    if not args.large_scale:
        all_psnr = all_psnr[:,:3900]
        all_bitrate = all_bitrate[:,:3900]
        all_dect = all_dect[:,:3900]
    # print('Segment shape:',all_psnr.shape)
    mean_psnr = np.sort(all_psnr.mean(axis=1)).tolist()
    mean_bpp = np.sort(all_bitrate.mean(axis=1)/pix_per_sec).tolist()
    # print('Mean PSNR:',np.sort(all_psnr.mean(axis=1)).tolist())
    # print('Mean bitrate:',np.sort(all_bitrate.mean(axis=1)).tolist())
    # print('Mean decoding time:',np.sort(all_dect.mean(axis=1)).tolist())
    return all_psnr,all_bitrate,all_dect,mean_psnr,mean_bpp

def simulate_over_traces(all_psnr,all_bitrate,all_dect,downthrpt,latency,sim_idx):
    # T_k = 1
    # Q_next = max(Q-T_k/p) + 1
    p = 1.*(GOP/args.fps) # seconds per segment
    # how to derive bola parameters from S1,S2,v1,v2,v_max,Q_low,Q_max
    avail_bitrates = all_bitrate.mean(axis=-1)
    avail_bitrates.sort()
    avail_psnr = all_psnr.mean(axis=-1)
    avail_psnr.sort()
    Q_max,Q_low = args.Q_max,args.Q_low
    S1 = avail_bitrates[0] * p
    S2 = avail_bitrates[1] * p
    if args.psnr:
        v1 = avail_psnr[0]
        v2 = avail_psnr[1]
        v_max = avail_psnr[-1]
    else:
        v1 = 0
        v2 = np.log(S2/S1)
        v_max = np.log(avail_bitrates[-1] * p / S1)
    alpha = (S1 * v2 - S2 * v1)/(S2 - S1)
    V = (Q_max - Q_low) / (v_max + alpha)
    gamma = (v_max * Q_low + alpha * Q_max) / (Q_max - Q_low) / p
    # print(f'V:{V}, gamma:{gamma}, v_max:{v_max}')
    # V = Q_max / (v_max + gamma * p)
    # x = [[i for i in range(Q_max+1)] for _ in range(5)]
    # y = [[] for _ in range(5)]
    # for bitrate in range(5):
    #     for curr_Q in range(Q_max+1):
    #         rho = (V * np.log(bitrate+1) + V * gamma * p - curr_Q)/((bitrate+1) * 1e6 * p)
    #         y[bitrate] += [rho]
    # line_plot(x,y,range(5),colors,'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/test.eps','Rho','Buffer (s)')  
    # y = np.array(y)
    # print(np.argmax(y,axis=0))

    # decoding has a separate queue to consume segments after each segment is downloaded
    num_levels = len(all_psnr)
    num_segments = all_bitrate.shape[1]
    curr_seg = 0 # try to download this segment
    download_finish_time = 0 # current time in seconds
    curr_Q = 0 # current buffer level in seconds, virtual buffer
    curr_real_Q = 0 # real buffer with frames
    bitstream_Q = 0
    remain_segments = num_segments
    mean_quality = 0
    selection_psnr = [[] for i in range(1+num_levels)]
    selection_bitrate = [[] for i in range(1+num_levels)]
    decode_finish_time = 0
    freq = 0
    while remain_segments > 0:
        # decide bitrate based on buffer level
        rho_max = -1000
        selected_level = -1
        curr_seg = num_segments - remain_segments
        rho_list = []
        for lvl in range(num_levels):
            if args.psnr:
                rho = (V * all_psnr[lvl,curr_seg] + V * gamma * p - curr_Q)/(all_bitrate[lvl,curr_seg] * p)
            else:
                rho = (V * np.log(all_bitrate[lvl,curr_seg]/S1) + V * gamma * p - curr_Q)/(all_bitrate[lvl,curr_seg] * p)
            rho_list += [rho]
            if rho >= 0 and rho > rho_max:
                rho_max = rho
                selected_level = lvl
        selection_psnr[selected_level] += [all_psnr[selected_level,curr_seg] if selected_level >= 0 else 0]
        selection_bitrate[selected_level] += [all_bitrate[selected_level,curr_seg] if selected_level >= 0 else 0]
        if selected_level == -1:
            # wait until buffer is enough
            # consume buffer
            if args.psnr:
                delta_T = curr_Q - (V * all_psnr[:,curr_seg].max() + V * gamma * p)
                curr_Q = V * all_psnr[:,curr_seg].max() + V * gamma * p
            else:
                delta_T = curr_Q - (V * np.log(all_bitrate[:,curr_seg].max()/S1) + V * gamma * p)
                curr_Q = V * np.log(all_bitrate[:,curr_seg].max()/S1) + V * gamma * p
            # accumulate time
            download_finish_time += delta_T
        else:
            # download segment based on bandwidth
            start_t = download_finish_time
            # size of a segment
            remain_size = all_bitrate[selected_level,curr_seg] * p
            # print('Trying to download size (bits):',remain_size)
            while remain_size > 0:
                # find current bandwidth
                trace_idx = int(download_finish_time/args.trace_dur)
                trace_end = (trace_idx+1)*args.trace_dur
                # print('Using bandwidth (bps):',downthrpt[trace_idx])
                # transmission + propagation
                downloadable = (trace_end - download_finish_time) * downthrpt[trace_idx]
                if downloadable >= remain_size:
                    download_finish_time += remain_size / downthrpt[trace_idx]
                    remain_size = 0
                else:
                    download_finish_time = trace_end
                    remain_size -= downloadable
            download_finish_time += latency[trace_idx]
            # accumulate time
            delta_T = download_finish_time - start_t
            # print('Time to download (s):',delta_T)
            # reduce segments
            remain_segments -= 1
            # add to quality
            mean_quality += all_psnr[selected_level,curr_seg]
            # decoding
            last_decT = decode_finish_time
            decode_finish_time = max(decode_finish_time,download_finish_time) + all_dect[selected_level,curr_seg]
            if curr_real_Q < (decode_finish_time - last_decT):
                freq += 1
            # consume and add to buffer
            curr_Q = p + max(curr_Q - (decode_finish_time - last_decT), 0)

            # if all_dect[selected_level,curr_seg] < p:
            #     curr_Q = p + max(curr_Q - (download_finish_time - start_t), 0)
            # else:
            #     curr_Q = p + max(curr_Q - (download_finish_time - start_t)*p/all_dect[selected_level,curr_seg], 0)
            curr_real_Q = p + max(curr_real_Q - (decode_finish_time - last_decT), 0)
        # print(sim_idx,'#Remain:',remain_segments,'Selected level',selected_level,'Buffer level (s):',curr_Q,'Current time:',download_finish_time,'finish:',decode_finish_time,num_segments * p)
    mean_bw = 0
    for tidx in range(0,trace_idx+1):
        mean_bw += downthrpt[trace_idx]
    mean_bw /= trace_idx+1
    # replay what is left in buffer
    finish_time = decode_finish_time + curr_Q
    # if all_dect[0,0]<p:
    #     finish_time = decode_finish_time + curr_Q
    # else:
    #     finish_time = decode_finish_time + curr_Q * all_dect[0,0] / p
    # rebuffering
    rebuffer_ratio = finish_time / (num_segments * p) - 1
    rebuffer_freq = freq / num_segments
    # quality
    mean_quality /= num_segments
    # QOE
    QoE = mean_quality - gamma * rebuffer_ratio
    # print(f'QoE:{QoE}, quality:{mean_quality}, rebuffer_ratio:{rebuffer_ratio}, mean bw usage:{mean_bw}')
    # print('Selection_counter:',[len(level) for level in selection_psnr])
    # print('Mean PSNR in each selection',[sum(level)/len(level) if level else 0 for level in selection_psnr])
    # print('Mean bitrate in each selection',[sum(level)/len(level) if level else 0 for level in selection_bitrate])
    return QoE,mean_quality,rebuffer_ratio,rebuffer_freq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--hardware', type=str, default='1080', help='3090,2080,1080')
    parser.add_argument("--trace_id", type=int, default=0, help="0:curr_videostream,1:curr_httpgetmt")
    parser.add_argument("--Q_max", type=int, default=60, help="Max buffer")
    parser.add_argument("--Q_low", type=int, default=10, help="Low buffer")
    parser.add_argument("--fps", type=int, default=30, help="frame per second")
    parser.add_argument('--psnr', dest='psnr', action='store_true')
    parser.add_argument('--no-psnr', dest='psnr', action='store_false')
    parser.set_defaults(psnr=False)
    parser.add_argument('--large_scale', dest='large_scale', action='store_true')
    parser.add_argument('--no-large_scale', dest='large_scale', action='store_false')
    parser.set_defaults(large_scale=True)
    parser.add_argument("--num_traces", type=int, default=1000, help="Number of traces.")
    args = parser.parse_args()
    
    if args.large_scale:
        BOLA_simulation(total_traces=args.num_traces)
    else:    
        BOLA_simulation(total_traces=args.num_traces,
            tasks=['LSVC-L-128','RLVC2','x264-veryfast','x264-medium','x264-veryslow','x265-veryfast','x265-medium','x265-veryslow'])