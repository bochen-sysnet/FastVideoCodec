import argparse
import re
import numpy as np

# fix gamma_p, change V and max buffer size (0-70s)
# bola-basic
# γ corresponds to how strongly we want to avoid rebuffering
# V buffer-perf metrics trade-off
# Q: buffer size
# set γp = 5 and varied V for different buffer sizes.
def BOLA_simulation(seed=0):
    with open(f'../{args.task}.log','r') as f:
        l = f.readlines()[0]
        l = l.strip()
        l = re.split('[\]\[]',l)
        l = [line for line in l if len(line)>0]
        summary = l[::3]
        detail = l[1::3]
    num_videos = 7
    num_levels = 4
    GOP = 13
    FPS = 30
    width,height = 960,640
    pix_per_frame = width*height
    pix_per_seg = pix_per_frame*GOP
    pix_per_sec = pix_per_frame*FPS
    all_psnr = []
    all_bitrate = []
    all_dect = []
    for vid in range(num_videos):
        seg_psnr = []
        seg_bitrate = []
        seg_dect = []
        for lid in range(num_levels):
            line_idx = vid + num_videos * lid
            psnr_list = detail[line_idx].split(',')
            num_seg = len(psnr_list)//GOP
            level_psnr = np.array(psnr_list[:num_seg*GOP]).astype(float).reshape(num_seg,GOP).mean(axis=-1).tolist()
            seg_psnr += [level_psnr]
            metrics_list = summary[line_idx].split(',')
            bpp,enct,dect = float(metrics_list[1]),float(metrics_list[2]),float(metrics_list[3])
            seg_bitrate += [[bpp * pix_per_sec] * num_seg]
            seg_dect += [[dect * GOP] * num_seg]
        seg_psnr = np.array(seg_psnr)
        seg_bitrate = np.array(seg_bitrate)
        seg_dect = np.array(seg_dect)
        all_psnr += [seg_psnr]
        all_bitrate += [seg_bitrate]
        all_dect += [seg_dect]
    all_psnr = np.concatenate(all_psnr,axis=1)
    all_bitrate = np.concatenate(all_bitrate,axis=1)
    all_dect = np.concatenate(all_dect,axis=1)
    num_segments = all_psnr.shape[1]
    print('Numer of segments:',num_segments)

    trace_dur = 1
    max_trace_num = num_segments*GOP/FPS/trace_dur*10
    # read network traces
    downthrpt = []
    latency = []
    import csv
    with open('../curr_videostream.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_cnt = 0
        for row in csv_reader:
            # random crop trace
            if line_cnt < seed * max_trace_num:
                line_cnt += 1
                continue
            # bytes per second to bps
            # micro seconds to sec
            if float(row["downthrpt"])>1e6:
                downthrpt += [float(row["downthrpt"])*8]
                latency += [float(row["latency"])/1e6]
            if len(downthrpt) >= max_trace_num:
                break
    print('Network trace loaded.')

    # T_k = 1
    # Q_next = max(Q-T_k/p) + 1
    p = 1.*(GOP/FPS) # seconds per segment
    Q_max,Q_low = 25,10
    # gamma = 5./(GOP/FPS)
    # V = 0.93
    # V = (Q_max - p) / (39 + gamma * p)
    # how to derive bola parameters from S1,S2,v1,v2,v_max,Q_low,Q_max
    avail_bitrates = all_bitrate.mean(axis=-1)
    avail_bitrates.sort()
    avail_psnr = all_psnr.mean(axis=-1)
    avail_psnr.sort()
    S1 = avail_bitrates[0] * p
    S2 = avail_bitrates[1] * p
    v1 = avail_psnr[0]
    v2 = avail_psnr[1]
    v_max = all_psnr.max(axis=-1)[0]
    alpha = (S1 * v2 - S2 * v1)/(S2 - S1)
    V = (Q_max - Q_low) / (v_max - alpha)
    gamma = (v_max * Q_low - alpha * Q_max) / (Q_max - Q_low) / p
    # less gamma less V

    curr_seg = 0 # try to download this segment
    curr_t = 0 # current time in seconds
    curr_Q = 0 # current buffer level in seconds
    remain_segments = num_segments
    mean_quality = 0
    selection_counter = [0 for i in range(1+num_levels)]
    while remain_segments > 0:
        # decide bitrate based on buffer level
        rho_max = -1000
        selected_level = -1
        curr_seg = num_segments - remain_segments
        for lvl in range(num_levels):
            rho = (V * all_psnr[lvl,curr_seg] + V * gamma * p - curr_Q)/(all_bitrate[lvl,curr_seg] * p)
            if rho >= 0 and rho > rho_max:
                rho_max = rho
                selected_level = lvl
        selection_counter[selected_level] += 1
        # print('Options:',all_psnr[:,curr_seg],all_bitrate[:,curr_seg])
        if selected_level == -1:
            # wait until buffer is enough
            delta_T = curr_Q - (V * all_psnr[:,curr_seg].max() + V * gamma * p)
            # accumulate time
            curr_t += delta_T
            # consume buffer
            curr_Q = V * all_psnr[:,curr_seg].max() + V * gamma * p
        else:
            # download segment based on bandwidth
            start_t = curr_t
            # size of a segment
            remain_size = all_bitrate[selected_level,curr_seg] * p
            # print('Trying to download size (bits):',remain_size)
            while remain_size > 0:
                # find current bandwidth
                trace_idx = int(curr_t/trace_dur)
                trace_end = (trace_idx+1)*trace_dur
                # print('Using bandwidth (bps):',downthrpt[trace_idx])
                # transmission + propagation + decoding
                downloadable = (trace_end - curr_t) * downthrpt[trace_idx] + latency[trace_idx] + all_dect[selected_level,curr_seg]
                if downloadable >= remain_size:
                    curr_t += remain_size / downthrpt[trace_idx]
                    remain_size = 0
                else:
                    curr_t = trace_end
                    remain_size -= downloadable
            # accumulate time
            delta_T = curr_t - start_t
            # print('Time to download (s):',delta_T)
            # reduce segments
            remain_segments -= 1
            # consume and add to buffer
            curr_Q = p + max(curr_Q - delta_T, 0)
            # add to quality
            mean_quality += all_psnr[lvl,curr_seg]
        if selected_level in [1,2]:
            print('#Remain:',remain_segments,'Selected level',selected_level,'Buffer level (s):',curr_Q)
    # replay what is left in buffer
    curr_t += curr_Q
    # rebuffering
    rebuffer_ratio = curr_t / (num_segments * p) - 1
    # quality
    mean_quality /= num_segments
    # QOE
    QoE = mean_quality - gamma * rebuffer_ratio
    print(f'QoE:{QoE},Quality:{mean_quality},rebuffer_ratio:{rebuffer_ratio}')
    print('Selection_counter',selection_counter)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters of simulations.')
    parser.add_argument('--task', type=str, default='x264-veryfast', help='RLVC,DVC,SPVC,AE3D,x265,x264')
    args = parser.parse_args()
    
    BOLA_simulation()
