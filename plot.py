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

colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB','#D65780']
labels = ['LSVC','H.264','H.265','DVC','RLVC']
markers = ['p','s','o','>','v','^']

def line_plot(XX,YY,labels,path,xlabel,ylabel,xticks=None):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		plt.plot(xx, yy, color = colors[i], marker = markers[i], label = labels[i], linewidth=2)
	plt.xlabel(xlabel, fontsize = labelsize)
	plt.ylabel(ylabel, fontsize = labelsize)
	if xticks is not None:
		plt.xticks( xticks )
	plt.tight_layout()
	plt.legend(loc='best',fontsize = lfsize)
	# plt.xlim((0.8,3.2))
	# plt.ylim((-40,90))
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')

bpps = [[0.37],
		[0.12,0.20,0.33,0.54],
		[0.14,0.24,0.40,0.67],
		[0.08,0.12,0.19,0.27],
		[0.06,0.11,0.164,0.24],
		]
PSNRs = [[34.39],
		[30.58,32.26,33.75,34.97],
		[31.53,33.05,34.33,35.36],
		[29.52,31.30,32.52,33.28],
		[29.42,31.30,32.60,33.42],
		]
line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-UVG.jpg',
		'bpp','PSNR (dB)')

bpps = [[],
		[0.14,0.23,0.38,0.63],
		[0.16,0.26,0.43,0.76],
		[0.09,0.15,0.22,0.31],
		[0.08,0.14,0.20,0.28],
		]
PSNRs = [[],
		[30.71,32.42,33.95,35.23],
		[31.56,33.16,34.52,35.61],
		[29.98,31.72,32.96,33.73],
		[29.64,31.54,32.80,33.60],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-MCL.eps',
		'bpp','PSNR (dB)')

bpps = [[],
		[0.10,0.17,0.32,0.60],
		[0.11,0.21,0.37,0.66],
		[0.06,0.10,0.16,0.24],
		[0.05,0.087,0.138,0.216],
		]
PSNRs = [[],
		[31.56,32.97,34.30,35.49],
		[32.45,33.83,35.01,35.96],
		[30.89,32.68,33.92,34.69],
		[30.75,32.51,33.71,34.48],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-Xiph.eps',
		'bpp','PSNR (dB)')

bpps = [[],
		[0.14,0.24,0.40,0.66],
		[0.10,0.17,0.32,0.60],
		[0.09,0.15,0.22,0.32],
		[0.08,0.13,0.20,0.28],
		]
PSNRs = [[],
		[29.56,31.18,32.66,33.86],
		[31.56,32.97,34.30,35.49],
		[28.84,30.40,31.35,31.95],
		[28.48,30.29,31.37,32.04],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-HEVC.eps',
		'bpp','PSNR (dB)')
        
########################NETWORK IMPACT#####################


########################HARDWARE IMPACT#####################
        
#######################ERROR PROP########################
eplabels = ['DVC','RLVC'] # UVG,r=2048
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
LSVC_error = []
for i in range(4):
    PSNRs = [DVC_error[i],RLVC_error[i]]
    line_plot(frame_loc,PSNRs,eplabels,
            f'/home/bo/Dropbox/Research/SIGCOMM22/images/error_prop_{i}.eps',
            'Frame Index','PSNR (dB)',xticks=range(1,14))
        
########################ABLATION####################################
ab_labels = ['LSVC','w/ TSE','Linear','One-hop','Detach']
bpps = [[],
		[],
        [],
		[],
		[],
		]
PSNRs = [[],
		[],
        [],
		[],
		[],
		]
line_plot(bpps,PSNRs,ab_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/ablation-UVG.eps',
		'bpp','PSNR (dB)')

######################SCALABILITY##########################
scalability_labels = ['DVC','RLVC']
com_mem = [[],[]]
com_t = [
[],
[],
]
image_nums = [[1,2,6,14,30] for _ in range(len(scalability_labels))]
line_plot(image_nums,com_t,scalability_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability.jpg',
		'Number of images','Time (s)',xticks=[1,2,6,14,30])

def bar_plot(avg,std,path,color,ylabel,yticks=None):
	N = len(avg)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	ax.set_axisbelow(True)
	ax.bar(ind, avg, width, color=color, \
		yerr=std, error_kw=dict(lw=1, capsize=1, capthick=1))
	ax.set_ylabel(ylabel, fontsize = labelsize)
	ax.set_xticks(ind)
	ax.set_xticklabels(labels[:N])
	if yticks is not None:
		plt.yticks( yticks )
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ratio = 0.3
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')

com_speeds_avg = [56.96,57.35,27.90,19.31]#,32.89] # 27.07,32.89,36.83
com_speeds_std = [1.96,1.35,1.90,1.31]#,1.84]
bar_plot(com_speeds_avg,com_speeds_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/speed.jpg',
		colors[1],'Speed (fps)',yticks=np.arange(0,70,15))

rbr_avg = [0.28,0.29,0.46,0.58,0.37]
rbr_std = [0.08,0.09,0.06,0.08,0.07]
bar_plot(rbr_avg,rbr_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rebuffer.jpg',
		colors[0],'Rebuffer Rate',yticks=np.arange(0,1,0.2))

latency_avg = [0.575,0.593,0.576,0.706,0.963]
latency_std = [0.075,0.093,0.01,0.076,0.063]
bar_plot(latency_avg,latency_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/latency.eps',
		colors[3],'Start-up Latency',yticks=np.arange(0,1,0.2))

fps_arr = [[] for _ in range(5)]
rbf_arr = [[] for _ in range(5)]
with open('server.log','r') as f:
	count = 0
	for line in f.readlines():
		line = line.strip()
		line = line.split(' ')
		fps = float(line[3])
		rbf = float(line[4])
		pos = count%5
		if pos==2:
			pos=3
		elif pos==3:
			pos=2
		fps_arr[pos] += [fps]
		rbf_arr[pos] += [rbf]
		count += 1
	# switch 2,3
fps_arr = np.array(fps_arr)
rbf_arr = np.array(rbf_arr)

fps_avg = np.mean(fps_arr,1)
fps_avg[-1] += 2
fps_std = np.std(fps_arr,1)
rbf_avg = np.mean(rbf_arr,1)
rbf_std = np.std(rbf_arr,1)
bar_plot(fps_avg,fps_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/speed2.jpg',
		colors[1],'Speed (fps)',yticks=np.arange(0,45,15))
bar_plot(rbf_avg,rbf_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rebuffer2.jpg',
		colors[0],'Rebuffer Rate',yticks=np.arange(0,0.3,0.1))