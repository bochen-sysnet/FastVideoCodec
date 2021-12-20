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

colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB']
labels = ['H.264','H.265','RLVC','DVC','LSVC']
markers = ['p','s','o','>','v']

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

bpps = [[0.14,0.21,0.33,0.5],
		[0.16,0.25,0.39,0.59],
		[0.12,0.187],
		[0.138,0.154],
		# [0.123,0.181,0.284,0.3925]
		]
PSNRs = [[28.37,29.96,31.31,32.38],
		[29.56,30.90,31.96,32.82],
		[26.19,30.82],
		[25.81,29.45],
		[28.98,30.54,31.54,32.24]]
line_plot(bpps,PSNRs,labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion.eps',
		'bpp','PSNR (dB)')
# 96
# [0.053132872000105635, 0.07760241199980555, 0.10680426199996873, 0.1364419829999406, 0.16576214699989578, 0.19565956099995674, 0.24735311800009185, 0.2696788140001445, 0.30823042800011535, 0.32221825099986745, 0.35182066400011536, 0.35663595400001213, 0.3842096920000131, 0.4138117239999701]
# 64
# [0.04606491999993523, 0.06779569900027127, 0.09685762100025386, 0.12219671000002563, 0.14586037300023236, 0.1700412850000248, 0.21632630700014488, 0.23113141200019527, 0.2638085839998894, 0.2760139719998733, 0.3001248149998901, 0.30320839800015165, 0.32136949599998843, 0.3438082170000598]

ml_labels = ['RLVC','DVC','LSVC']
com_t = [[0.031189141000140808, 0.061611389999825406, 0.08575277299974005, 0.11296145900018928, 0.14397867699995004, 0.1720223359998272, 0.20626382000023114, 0.23150906300020324, 0.262774699999909, 0.29222327299953577, 0.32226526900012686, 0.3528986520000217, 0.38183643099978326, 0.41159681199997067],
[0.03548506700008147, 0.07703001000004406, 0.11634391000029609, 0.1602356679998138, 0.20212238199974308, 0.24543897899980038, 0.28723739499992007, 0.33600276399988616, 0.3826641949999612, 0.42505351599993446, 0.4697764459995142, 0.5151504109999223, 0.5607268250005291, 0.6026666810003007],
[0.04606491999993523, 0.06779569900027127, 0.09685762100025386, 0.12219671000002563, 0.14586037300023236, 0.1700412850000248, 0.21632630700014488, 0.23113141200019527, 0.2638085839998894, 0.2760139719998733, 0.3001248149998901, 0.30320839800015165, 0.32136949599998843, 0.3438082170000598]
]
image_nums = [range(1,15) for _ in range(2)]
line_plot(image_nums,com_t,ml_labels,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability.eps',
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

com_speeds_avg = [56.96,57.35,19.31,27.90]#,32.89] # 27.07,32.89,36.83
com_speeds_std = [1.96,1.35,1.31,1.90]#,1.84]
bar_plot(com_speeds_avg,com_speeds_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/speed.eps',
		colors[1],'Speed (fps)',yticks=np.arange(0,70,15))

rbr_avg = [0.28,0.29,0.58,0.46,0.37]
rbr_std = [0.08,0.09,0.08,0.06,0.07]
bar_plot(rbr_avg,rbr_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rebuffer.eps',
		colors[4],'Rebuffer Rate',yticks=np.arange(0,1,0.2))

latency_avg = [0.575,0.593,0.706,0.576,0.963]
latency_std = [0.075,0.093,0.01,0.076,0.063]
bar_plot(latency_avg,latency_std,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/latency.eps',
		colors[3],'Start-up Latency',yticks=np.arange(0,1,0.2))