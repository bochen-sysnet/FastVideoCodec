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

eplabels = ['RC','MC','WP']
frame_loc = [[i for i in range(1,7)],
			[i for i in range(1,7)],
			[i for i in range(1,7)]]
PSNRs = [[34.90052795410156, 34.51676940917969, 34.32107162475586, 34.08355712890625, 34.34242630004883, 34.14742660522461],
[31.397937774658203, 31.620567321777344, 30.092472076416016, 28.11025619506836, 31.539072036743164, 29.98174476623535],
[30.749130249023438, 31.375137329101562, 29.74481964111328, 27.376392364501953, 31.241863250732422, 29.615171432495117]]

# PSNRs = [[34.137001037597656, 34.37543487548828, 34.10160446166992, 34.343231201171875, 34.55673599243164, 34.92389678955078, 33.563720703125, 34.90052795410156, 34.51676940917969, 34.32107162475586, 34.08355712890625, 34.34242630004883, 34.14742660522461],
# [32.98979568481445, 33.08745193481445, 33.179832458496094, 33.29349136352539, 33.42366027832031, 33.544105529785156, 33.563720703125, 33.54086685180664, 33.41876983642578, 33.29288864135742, 33.180362701416016, 33.07490921020508, 32.98408889770508],
		
line_plot(frame_loc,PSNRs,eplabels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/error_prop.eps',
		'Frame Index','PSNR (dB)',xticks=range(1,7))

# bpps = [[0.12,0.20,0.33,0.54],
# 		[0.14,0.24,0.40,0.67],
# 		[0.08,0.12,0.19,0.27],
# 		[0.06,0.10,0.16,0.22],
# 		[0.102,0.174,0.264,0.3889]
# 		]
# PSNRs = [[30.58,32.26,33.75,34.97],
# 		[31.53,33.05,34.33,35.36],
# 		[29.52,31.30,32.52,33.28],
# 		[29.32,30.89,32.15,32.74],
# 		[28.84,30.41,31.46,32.09]
# 		]

bpps = [[],
		[0.12,0.20,0.33,0.54],
		[0.14,0.24,0.40,0.67],
		[0.08,0.12,0.19,0.27],
		[0.06,0.11,0.165,0.24],
		]
PSNRs = [[],
		[30.58,32.26,33.75,34.97],
		[31.53,33.05,34.33,35.36],
		[29.52,31.30,32.52,33.28],
		[29.42,31.17,32.40,33.22],
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
		[29.64,31.42,32.61,33.31],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-MCL.eps',
		'bpp','PSNR (dB)')

bpps = [[],
		[0.10,0.17,0.32,0.60],
		[0.11,0.21,0.37,0.66],
		[0.06,0.10,0.16,0.24],
		[0.05,0.09,0.14,0.22],
		]
PSNRs = [[],
		[31.56,32.97,34.30,35.49],
		[32.45,33.83,35.01,35.96],
		[30.89,32.68,33.92,34.69],
		[30.75,32.46,33.63,34.35],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-Xiph.eps',
		'bpp','PSNR (dB)')

bpps = [[],
		[0.14,0.24,0.40,0.66],
		[0.10,0.17,0.32,0.60],
		[],
		[],
		]
PSNRs = [[],
		[29.56,31.18,32.66,33.86],
		[31.56,32.97,34.30,35.49],
		[],
		[],
		]

line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-HEVC.eps',
		'bpp','PSNR (dB)')
#################################################################################


ab_labels = ['Base','C64','C128','Recurrent','Detach','Linear']
bpps = [[],
		[0.102,0.174,0.264,0.3889],
		[0.418],
		[0.123,0.181,0.284,0.3925],
		[0.25],
		[],
		]
PSNRs = [[],
		[28.84,30.41,31.46,32.09],
		[30.93],
		[28.98,30.54,31.54,32.24],
		[30.83],
		[],
		]
line_plot(bpps,PSNRs,ab_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/ablation-UVG.eps',
		'bpp','PSNR (dB)')

# 96
# [0.053132872000105635, 0.07760241199980555, 0.10680426199996873, 0.1364419829999406, 0.16576214699989578, 0.19565956099995674, 0.24735311800009185, 0.2696788140001445, 0.30823042800011535, 0.32221825099986745, 0.35182066400011536, 0.35663595400001213, 0.3842096920000131, 0.4138117239999701]
# 64
# [0.04606491999993523, 0.06779569900027127, 0.09685762100025386, 0.12219671000002563, 0.14586037300023236, 0.1700412850000248, 0.21632630700014488, 0.23113141200019527, 0.2638085839998894, 0.2760139719998733, 0.3001248149998901, 0.30320839800015165, 0.32136949599998843, 0.3438082170000598]

ml_labels = ['DVC','RLVC','LSVC']
com_t = [[0.031189141000140808, 0.061611389999825406, 0.08575277299974005, 0.11296145900018928, 0.14397867699995004, 0.1720223359998272, 0.20626382000023114, 0.23150906300020324, 0.262774699999909, 0.29222327299953577, 0.32226526900012686, 0.3528986520000217, 0.38183643099978326, 0.41159681199997067],
[0.03548506700008147, 0.07703001000004406, 0.11634391000029609, 0.1602356679998138, 0.20212238199974308, 0.24543897899980038, 0.28723739499992007, 0.33600276399988616, 0.3826641949999612, 0.42505351599993446, 0.4697764459995142, 0.5151504109999223, 0.5607268250005291, 0.6026666810003007],
[0.0538, 0.0816, 0.107, 0.136, 0.167, 0.194, 0.245, 0.266, 0.311, 0.322, 0.352, 0.354, 0.387, 0.413]
]
image_nums = [range(1,15) for _ in range(2)]
line_plot(image_nums,com_t,ml_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability.jpg',
		'Number of images','Time (s)',xticks=[0,5,10,15])

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