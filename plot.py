#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
lfsize = 18
labelsize = 24
linewidth = 4
mksize = 4
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
# colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB','#D65780']
labels = ['LSVC','H.264','H.265','DVC','RLVC']
markers = ['p','s','o','>','v','^']

def line_plot(XX,YY,label,color,path,xlabel,ylabel,
				xticks=None,yticks=None,ncol=None, yerr=None,
				use_arrow=False,arrow_coord=(0.4,30)):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], label = label[i], linewidth=2)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], marker = markers[i], label = label[i], linewidth=2)
	plt.xlabel(xlabel, fontsize = labelsize)
	plt.ylabel(ylabel, fontsize = labelsize)
	if xticks is not None:
		plt.xticks( xticks )
	if yticks is not None:
		plt.yticks(yticks)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=15,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	plt.tight_layout()
	if ncol is None:
		plt.legend(loc='best',fontsize = lfsize)
	else:
		plt.legend(loc='best',fontsize = lfsize,ncol=ncol)
	# plt.xlim((0.8,3.2))
	# plt.ylim((-40,90))
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.clf()

def bar_plot(avg,std,label,path,color,ylabel,yticks=None):
	N = len(avg)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	ax.set_axisbelow(True)
	hbar = ax.bar(ind, avg, width, color=color, \
		yerr=std, error_kw=dict(lw=1, capsize=1, capthick=1))
	ax.set_ylabel(ylabel, fontsize = labelsize)
	ax.set_xticks(ind)
	ax.set_xticklabels(label)
	ax.bar_label(hbar, fmt='%.2f', fontsize = labelsize)
	if yticks is not None:
		plt.yticks( yticks )
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ratio = 0.3
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')

Ubpps = [[0.12,0.18,0.266,0.37],
		[0.12,0.20,0.33,0.54],
		[0.14,0.24,0.40,0.67],
		[0.08,0.12,0.19,0.27],
		[0.06,0.11,0.164,0.24],
		]
UPSNRs = [[30.63,32.17,33.52,34.39],
		[30.58,32.26,33.75,34.97],
		[31.53,33.05,34.33,35.36],
		[29.52,31.30,32.52,33.28],
		[29.42,31.30,32.60,33.42],
		]
Ubpps = np.array(Ubpps)
UPSNRs = np.array(UPSNRs)
line_plot(Ubpps,UPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-UVG.eps',
		'bpp','PSNR (dB)')

line_plot(Ubpps[1:],UPSNRs[1:],labels[1:],colors[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation0.eps',
		'bpp','PSNR (dB)',use_arrow=True,arrow_coord=(0.1,34))

Mbpps = [[0.14,0.21,0.30,0.41],
		[0.14,0.23,0.38,0.63],
		[0.16,0.26,0.43,0.76],
		[0.09,0.15,0.22,0.31],
		[0.08,0.14,0.20,0.28],
		]
MPSNRs = [[30.93,32.47,33.75,34.57],
		[30.71,32.42,33.95,35.23],
		[31.56,33.16,34.52,35.61],
		[29.98,31.72,32.96,33.73],
		[29.64,31.54,32.80,33.60],
		]

line_plot(Mbpps,MPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-MCL.eps',
		'bpp','PSNR (dB)')

Xbpps = [[0.10,0.147,0.22,0.34],
		[0.10,0.17,0.32,0.60],
		[0.11,0.21,0.37,0.66],
		[0.06,0.10,0.16,0.24],
		[0.05,0.087,0.138,0.216],
		]
XPSNRs = [[31.84,33.24,34.51,35.40],
		[31.56,32.97,34.30,35.49],
		[32.45,33.83,35.01,35.96],
		[30.89,32.68,33.92,34.69],
		[30.75,32.51,33.71,34.48],
		]

line_plot(Xbpps,XPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-Xiph.eps',
		'bpp','PSNR (dB)')

Hbpps = [[0.14,0.21,0.307,0.37],
		[0.14,0.24,0.40,0.66],
		[0.16,0.275,0.47,0.77],
		[0.09,0.15,0.22,0.32],
		[0.08,0.13,0.20,0.28],
		]
HPSNRs = [[29.75,31.28,32.39,33.05],
		[29.56,31.18,32.66,33.86],
		[30.50,32.01,33.28,34.24],
		[28.84,30.40,31.35,31.95],
		[28.48,30.29,31.37,32.04],
		]

line_plot(Hbpps,HPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-HEVC.eps',
		'bpp','PSNR (dB)')
        
#######################ERROR PROP########################
eplabels = ['LSVC','DVC','RLVC'] # UVG,r=2048
frame_loc = [[i for i in range(1,14)] for _ in range(len(eplabels))]
DVC_error = [
[29.17373275756836, 29.27086639404297, 29.370689392089844, 29.497406005859375, 29.661972045898438, 29.852237701416016, 30.150089263916016, 29.852378845214844, 29.661556243896484, 29.503952026367188, 29.374088287353516, 29.260000228881836, 29.161962509155273],
[30.87851333618164, 30.991844177246094, 31.11360740661621, 31.27014923095703, 31.462120056152344, 31.68387794494629, 32.09651565551758, 31.678407669067383, 31.453895568847656, 31.26244354248047, 31.11084747314453, 30.96697235107422, 30.851476669311523],
[32.132972717285156, 32.243431091308594, 32.35715866088867, 32.50340270996094, 32.68071746826172, 32.86759948730469, 33.17623519897461, 32.866546630859375, 32.67479705810547, 32.504638671875, 32.3599967956543, 32.226200103759766, 32.11112594604492],
[32.98991012573242, 33.08746337890625, 33.17982864379883, 33.293479919433594, 33.42365646362305, 33.544105529785156, 33.5637092590332, 33.54086685180664, 33.418758392333984, 33.292877197265625, 33.18037414550781, 33.075035095214844, 32.984283447265625],
]
RLVC_error = [
[28.934640884399414, 29.0654354095459, 29.188518524169922, 29.363454818725586, 29.605087280273438, 30.001672744750977, 30.150089263916016, 29.993770599365234, 29.60120391845703, 29.366622924804688, 29.19466209411621, 29.043128967285156, 28.91464614868164],
[30.781314849853516, 30.918954849243164, 31.047901153564453, 31.23015785217285, 31.494022369384766, 31.950214385986328, 32.09651565551758, 31.95067596435547, 31.49615478515625, 31.236337661743164, 31.052120208740234, 30.89519691467285, 30.753807067871094],
[32.05003356933594, 32.206790924072266, 32.35562515258789, 32.55833435058594, 32.83184051513672, 33.269229888916016, 33.17623519897461, 33.277217864990234, 32.842647552490234, 32.578739166259766, 32.3740234375, 32.1959228515625, 32.042484283447266],
[32.94842529296875, 33.0999755859375, 33.23988723754883, 33.43804168701172, 33.688114166259766, 34.05646896362305, 33.5637092590332, 34.05782699584961, 33.69127655029297, 33.450496673583984, 33.25787353515625, 33.094661712646484, 32.95338439941406],
]
LSVC_error = [
[30.324174880981445, 30.538787841796875, 30.67068862915039, 30.49388313293457, 30.715848922729492, 31.25832176208496, 30.150089263916016, 31.260454177856445, 30.7042293548584, 30.511245727539062, 30.683881759643555, 30.523727416992188, 30.34703254699707],
[31.795574188232422, 32.01017379760742, 32.13482666015625, 32.016048431396484, 32.25210189819336, 32.848426818847656, 32.09651565551758, 32.85036849975586, 32.24248123168945, 32.039222717285156, 32.14407730102539, 31.9993953704834, 31.81879997253418],
[33.16758728027344, 33.39623260498047, 33.38203430175781, 33.413753509521484, 33.68014144897461, 34.21955108642578, 33.17623519897461, 34.22688674926758, 33.68741989135742, 33.44963455200195, 33.401241302490234, 33.39918518066406, 33.20851516723633],
[34.112457275390625, 34.34897232055664, 34.174808502197266, 34.358848571777344, 34.64569854736328, 35.04032516479492, 33.5637092590332, 35.05060958862305, 34.66664505004883, 34.40544891357422, 34.20201110839844, 34.366241455078125, 34.17375183105469],
]
for i in range(4):
    PSNRs = [LSVC_error[i],DVC_error[i],RLVC_error[i]]
    line_plot(frame_loc,PSNRs,eplabels,colors,
            f'/home/bo/Dropbox/Research/SIGCOMM22/images/error_prop_{i}.eps',
            'Frame Index','PSNR (dB)',xticks=range(1,14))
        
########################ABLATION####################################
# UVG
ab_labels = ['LSVC','w/o TSE','Linear','One-hop']
bpps = [[0.12,0.18,0.266,0.37],
		[0.12,0.20,0.30,0.41],
        [0.10,0.15,0.23,0.33],
		[0.11,0.19,0.28,],
		]
PSNRs = [[30.63,32.17,33.52,34.39],
		[29.83,31.25,32.74,34.05],
        [29.33,31.15,32.59,33.65],
		[29.77,31.50,32.94,],
		]
line_plot(bpps,PSNRs,ab_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/ablation_e.eps',
		'bpp','PSNR (dB)')

# speed
fps_avg_list = []
fps_std_list = []
with open('ablation.log','r') as f:
	count = 0
	fps_arr = []
	for idx,line in enumerate(f.readlines()):
		line = line.strip()
		line = line.split(' ')
		fps_arr += [float(line[3])]
		if idx%4==3:
			fps_arr = np.array(fps_arr)
			fps_avg,fps_std = np.mean(fps_arr),np.std(fps_arr)
			fps_avg_list.append(fps_avg)
			fps_std_list.append(fps_std)
			fps_arr = []
ab_labels = ['LSVC','w/o TSE','Linear','One-hop']
bar_plot(fps_avg_list,fps_std_list,ab_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/ablation_s.eps',
		'#4f646f','Speed (fps)')

######################SCALABILITY##########################
# motivation show duration
scalability_labels = ['LSVC','DVC','RLVC']
# read
fps_avg_list = []
fps_std_list = []
gpu_avg_list = []
gpu_std_list = []
with open('scalability.log','r') as f:
	count = 0
	fps_arr = []
	gpu_arr = []
	for idx,line in enumerate(f.readlines()):
		line = line.strip()
		line = line.split(' ')
		fps_arr += [float(line[3])]
		gpu_arr += [float(line[6])]
		if idx%4==3:
			fps_arr = np.array(fps_arr)
			gpu_arr = np.array(gpu_arr)/8117
			fps_avg,fps_std = np.mean(fps_arr),np.std(fps_arr)
			gpu_avg,gpu_std = np.mean(gpu_arr),np.std(gpu_arr)
			fps_avg_list.append(fps_avg)
			fps_std_list.append(fps_std)
			gpu_avg_list.append(gpu_avg)
			gpu_std_list.append(gpu_std)
			fps_arr = []
			gpu_arr = []

fps_avg_list = np.array(fps_avg_list)
fps_avg_list.resize(len(scalability_labels),30)
fps_std_list = np.array(fps_std_list)
fps_std_list.resize(len(scalability_labels),30)
gpu_avg_list = np.array(gpu_avg_list)
gpu_avg_list.resize(len(scalability_labels),30)
gpu_std_list = np.array(gpu_std_list)
gpu_std_list.resize(len(scalability_labels),30)

show_indices = [0,1,5,13,29] # 1,2,6,14,30
GOP_size = [[i+2 for i in show_indices] for _ in range(len(scalability_labels))]
line_plot(GOP_size,fps_avg_list[:,show_indices],scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_fps.eps',
		'GOP Size','Coding Speed (fps)',ncol=len(scalability_labels),yerr=fps_std_list[:,show_indices],
		yticks=range(10,50,10),xticks=range(5,31,5))
line_plot(GOP_size,gpu_avg_list[:,show_indices],scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_gpu.eps',
		'GOP Size','GPU Usage (%)',xticks=range(5,31,5),yerr=gpu_std_list[:,show_indices])

SPSNRs = [
[30.91,32.62,33.89,34.57],
[30.94,32.58,33.87,34.60],
[30.63,32.17,33.52,34.39],
[30.17,31.72,33.12,34.07],
[29.72,31.29,32.74,33.76],
]
Sbpps = [
[0.23,0.36,0.54,0.74],
[0.21,0.30,0.44,0.61],
[0.12,0.18,0.266,0.37],
[0.11,0.16,0.22,0.31],
[0.10,0.15,0.21,0.30],
]
sc_labels = ['GOP=3','GOP=5','GOP=13 (Our setup)','GOP=29','GOP=61']
line_plot(Sbpps,SPSNRs,sc_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_e.eps',
		'bpp','PSNR (dB)')

# motiv
show_indices = range(30)
GOP_size = [[i+2 for i in show_indices] for _ in range(2)]
line_plot(GOP_size,gpu_avg_list[1:,show_indices],scalability_labels[1:],colors[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation2.eps',
		'GOP Size','GPU Usage (%)',xticks=range(5,31,5))
show_indices = range(30)#[0,1,5,13,29]
GOP_size = [[i+2 for i in show_indices] for _ in range(3)]
total_time = 1/fps_avg_list[:,show_indices]*(1+np.array(show_indices))
line_plot(GOP_size,total_time,scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation3.eps',
		'GOP Size','Coding Duration (s)',xticks=range(5,31,5))
# result show fps



########################HARDWARE IMPACT#####################
# RTX2080,RTX2070
# all hardware in same plot: x(HW),y(speed)

# GTX1080 performance
# encoder only
# encoding speed
fps_avg_list = []
fps_std_list = []
with open('1080_speed.log','r') as f:
	count = 0
	fps_arr = []
	for idx,line in enumerate(f.readlines()):
		line = line.strip()
		line = line.split(' ')
		fps_arr += [float(line[3])]
		if idx%4==3:
			fps_arr = np.array(fps_arr)
			fps_avg,fps_std = np.mean(fps_arr),np.std(fps_arr)
			fps_avg_list.append(fps_avg)
			fps_std_list.append(fps_std)
			fps_arr = []

bar_plot(fps_avg_list,fps_std_list,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/speed.eps',
		colors[1],'Speed (fps)')

# bar_plot(fps_avg_list[1:],fps_std_list[1:],labels[1:],
# 		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation1.eps',
# 		'#4f646f','Coding Speed (fps)',yticks=[30,500])
def hbar_plot(avg,std,label,path,color,xlabel,xticks=None):
	plt.rcdefaults()
	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)

	y_pos = np.arange(len(avg))
	width = 0.5
	hbars1 = ax1.barh(y_pos, avg, width, color=color, xerr=std, align='center', error_kw=dict(lw=1, capsize=1, capthick=1))
	hbars2 = ax2.barh(y_pos, avg, width, color=color, xerr=std, align='center', error_kw=dict(lw=1, capsize=1, capthick=1))
	
	ax1.set_xlim(0,200)
	ax2.set_xlim(450,500)

	# hide the spines between ax and ax2
	ax1.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	# ax1.tick_params(labelright='off')

	d = .015 # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='r', clip_on=False)
	ax1.plot((1-d,1+d), (-d,+d), **kwargs)
	ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d,+d), (1-d,1+d), **kwargs)
	ax2.plot((-d,+d), (-d,+d), **kwargs)

	ax1.bar_label(hbars1, fmt='%.2f')
	ax2.bar_label(hbars2, fmt='%.2f')
	ax1.set_yticks(y_pos, labels=label)
	ax1.invert_yaxis()  
	plt.tight_layout()
	fig.text(0.5, 0, xlabel, ha='center')
	fig.savefig(path,bbox_inches='tight')
hbar_plot(fps_avg_list[1:],fps_std_list[1:],labels[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation1.eps',
		'#4f646f','Coding Speed (fps)',xticks=[30,150,500])

########################NETWORK IMPACT#####################
# FPS,Rebuffer,Latency
def get_mean_std_from(pos,filename):
	arr = [[[] for _ in range(4)] for _ in range(5)]
	with open(filename,'r') as f:
		count = 0
		for line in f.readlines():
			line = line.strip()
			line = line.split(' ')
			v = float(line[pos])
			i = (count%20)//4 # method
			j = (count%20)%4 # lambda value
			arr[i][j] += [v]
			count += 1
	arr = np.array(arr)
	arr.resize(5,4*len(arr[0][0]))
	avg = np.mean(arr,1)
	std = np.std(arr,1)
	return avg,std

def get_arr_from(pos,filename):
	arr = [[[] for _ in range(4)] for _ in range(5)]
	with open(filename,'r') as f:
		count = 0
		for line in f.readlines():
			line = line.strip()
			line = line.split(' ')
			v = float(line[pos])
			i = (count%20)//4 # method
			j = (count%20)%4 # lambda value
			arr[i][j] += [v]
			count += 1
	arr = np.array(arr)
	return arr

# NET 1
fps_arr = get_arr_from(3,'live_client.log')
fps_arr = np.mean(fps_arr,2)

k = 0
for psnr,bpp in [(UPSNRs,Ubpps),(MPSNRs,Mbpps),(XPSNRs,Xbpps),(HPSNRs,Hbpps)]:
	throughput = np.array(bpp)/fps_arr
	# used to compute throughput
	line_plot(throughput,psnr,labels,colors,
		f'/home/bo/Dropbox/Research/SIGCOMM22/images/bpep-distortion_{k}.eps',
		'bpep','PSNR (dB)')
	k += 1

fps_avg,fps_std = get_mean_std_from(3,'live_client.log')
rbf_avg,rbf_std = get_mean_std_from(4,'live_server.log')
lat_avg,lat_std = get_mean_std_from(5,'live_server.log')

bar_plot(fps_avg,fps_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/framerate.eps',
		colors[0],'Frame Rate',yticks=range(0,40,10))
bar_plot(rbf_avg,rbf_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rebuffer.eps',
		colors[2],'Rebuffer Rate',yticks=[0,0.1,.2,.3])
bar_plot(lat_avg,lat_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/latency.eps',
		colors[4],'Start-up Latency',yticks=[0,0.5,1,1.5])

# NET 2
fps_arr = get_arr_from(3,'lossy_client.log')
fps_arr = np.mean(fps_arr,2)

k = 0
for psnr,bpp in [(UPSNRs,Ubpps),(MPSNRs,Mbpps),(XPSNRs,Xbpps),(HPSNRs,Hbpps)]:
	throughput = np.array(bpp)/fps_arr
	# used to compute throughput
	line_plot(throughput,psnr,labels,colors,
		f'/home/bo/Dropbox/Research/SIGCOMM22/images/bpep-distortion_lossy_{k}.eps',
		'bpep','PSNR (dB)')
	k += 1

fps_avg,fps_std = get_mean_std_from(3,'lossy_client.log')
rbf_avg,rbf_std = get_mean_std_from(4,'lossy_server.log')
lat_avg,lat_std = get_mean_std_from(5,'lossy_server.log')

bar_plot(fps_avg,fps_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/framerate_loss.eps',
		colors[0],'Frame Rate',yticks=range(0,40,10))
bar_plot(rbf_avg,rbf_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rebuffer_loss.eps',
		colors[2],'Rebuffer Rate',yticks=[0,0.1,.2,.3,.4])
bar_plot(lat_avg,lat_std,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/latency_loss.eps',
		colors[4],'Start-up Latency')