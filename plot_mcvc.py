#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import random, math

labelsize_b = 14
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#1C4670','#FF9636','#9D5FFB','#21B6A8','#D65780']
# colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
# colors =['#FD0707','#0D0DDF','#129114','#DDDB03','#FF8A12','#8402AD']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
colors = ["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a", "#fdbf6f", "#ff7f00"]
# colors = ["#006d2c", "#31a354", "#74c476", "#bae4b3", "#ececec", "#969696"]
colors = ["#004c6d", "#f18f01", "#81b214", "#c7243a", "#6b52a1", "#a44a3f"]

views_of_category = [4,6,5,4,4]
markers = ['s','o','^','v','D','<','>','P','*'] 
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
methods3 = ['Replication (Optimal)','Partition (Ours)','Standalone']
methods6 = ['Ours','Baseline','Optimal$^{(2)}$','Ours*','Baseline*','Optimal$^{(2)}$*']
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dashed',              (0, (5, 5))),
     ('dotted',              (0, (1, 5))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('densely dashed',      (0, (5, 1))),

     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]

from matplotlib.patches import Ellipse

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(60,0.6),markersize=16,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_probarrow=False,
				rotation=None,use_resnet56_2arrow=False,use_resnet56_3arrow=False,use_resnet56_4arrow=False,use_resnet50arrow=False,use_re_label=False,
				use_throughput_annot=False,use_connarrow=False,lgsize=None,oval=False,scatter_soft_annot=False,markevery=1,annot_aw=None,
				fill_uu=False,failure_annot=False,failure_repl_annot=False,use_dcnbw_annot=False,latency_infl_annot=False,ur_annot=False,
				markersize_list=[],markers=markers,markerfacecolor='none',display_annot=[],si_annot=False,sr_annot=False,
				saving_annot=None,mps_annot=False,ablation_annot=False,sisr_annot=False,bw_annot=False,si_annot2=False):
	if lgsize is None:
		lgsize = lbsize
	if get_ax==1:
		ax = plt.subplot(211)
	elif get_ax==2:
		ax = plt.subplot(212)
	else:
		fig, ax = plt.subplots()
	ax.grid(zorder=0)
	handles = []
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if logx:
			xx = np.log10(np.array(xx))
		if yerr is None:
			if not markersize_list:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery)
			else:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize_list[i], markevery=markevery)
		else:
			if markersize > 0:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					marker = markers[i], label = label[i], 
					linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery,
					capsize=4)
			else:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					label = label[i], 
					linewidth=linewidth,
					capsize=4)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lbsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lbsize)
	if xticklabel is not None:
		ax.set_xticklabels(xticklabel)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=45, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	if ablation_annot:
		ax.text(0.3,29,"1.8dB-4dB\nreduction", ha="center", va="center", size=lgsize,color=color[3],fontweight='bold')
		ax.text(1,32,"3.75X bitrates", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.annotate(text="",xy=(0.4,31.66),xytext=(1.4,31.66), arrowprops=dict(arrowstyle='<-',lw=2),size=lbsize)
	if bw_annot:
		ax.text(0.55,85.5,"Higher rec. quality,\nbetter BW saving", ha="center", va="center", size=lgsize,fontweight='bold')
	if si_annot2:
		ax.text(0.27,230,"Higher rec. quality,\nless sampled frames", ha="center", va="center", size=lgsize,fontweight='bold')
	if sisr_annot:
		ax.text(-0.75,28,"Gain"+r'$\leq$'+"1dB\nif ratio"+r'$\leq 0.01$', ha="center", va="center", size=lgsize,fontweight='bold')
	if mps_annot:
		ax.text(22*16,29.5,"Full", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.text(12*16,29.5,"Half", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.text(5*16,33,"0.47dB-0.75dB\nloss with\nhalf cache", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.annotate(text="",xy=(160,28),xytext=(160,34), arrowprops=dict(arrowstyle='-',lw=2),size=lbsize)
		ax.annotate(text="",xy=(320,28),xytext=(320,34), arrowprops=dict(arrowstyle='-',lw=2),size=lbsize)
	if saving_annot is not None:
		c = color[4] if saving_annot[1]>1 else color[2]
		h = 0.7 if saving_annot[1]>1 else 0.9e-6
		ax.text((saving_annot[2]+saving_annot[3])/2-0.1,saving_annot[1]-h,u'\u2193'+f"{saving_annot[0]}%", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.annotate(text="",xy=(saving_annot[2],saving_annot[1]),xytext=(saving_annot[3],saving_annot[1]), arrowprops=dict(arrowstyle='->',lw=2,color=c),size=lbsize,fontweight='bold')
	if display_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.annotate(label[i], xy=(xx[0],yy[0]), xytext=(xx[0]+display_annot[i][0],yy[0]+display_annot[i][1]), fontsize=lbsize-4,)
	if si_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.text(xx[3],yy[3]+0.2, u'\u2713', ha="center", va="center", size=lbsize,fontweight='bold')
		ax.text(-1.5,33.3, u'\u2713'+" AcID's Config", ha="center", va="center", size=lbsize,fontweight='bold')
	if sr_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.text(xx[-1],yy[-1]-0.3, u'\u2713', ha="center", va="center", size=lbsize,fontweight='bold')
		ax.text(-0.6,27.4, u'\u2713'+" AcID's Config", ha="center", va="center", size=lbsize,fontweight='bold')
	if use_throughput_annot:
		ax.annotate(text=f"$\u2191$"+'41%', xy=(XX[1][1],YY[1][1]), xytext=(0,0.8), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize+2,fontweight='bold')
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lgsize)
		else:
			if bbox_to_anchor is None:
				plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol)
			else:
				if oval:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor, handles=handles)
				else:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	if get_ax!=0:
		return ax
	fig.savefig(path,bbox_inches='tight')
	plt.close()


def groupedbar(data_mean,data_std,ylabel,path,yticks=None,envs = [2,3,4],
				methods=['Ours','Standalone','Optimal','Ours*','Standalone*','Optimal*'],use_barlabel_x=False,use_barlabe_y=False,
				ncol=3,bbox_to_anchor=(0.46, 1.28),sep=1.25,width=0.15,xlabel=None,legloc=None,labelsize=labelsize_b,ylim=None,
				use_downarrow=False,rotation=None,lgsize=None,yticklabel=None,latency_annot=False,bandwidth_annot=False,latency_met_annot=False,
				showaccbelow=False,showcompbelow=False,bw_annot=False,showrepaccbelow=False,breakdown_annot=False,frameon=True,c2s_annot=False):
	if lgsize is None:
		lgsize = labelsize
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)*sep
	# colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


	ax.grid()
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['top'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	if rotation is None:
		plt.xticks(center_index, envs, size=labelsize)
	else:
		plt.xticks(center_index, envs, size=labelsize, rotation=rotation)
	plt.xticks(fontsize=labelsize)
	plt.yticks(fontsize=labelsize)
	ax.set_ylabel(ylabel, size=labelsize)
	if xlabel is not None:
		ax.set_xlabel(xlabel, size=labelsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=labelsize)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if ylim is not None:
		ax.set_ylim(ylim)
	for i in range(num_methods):
		x_index = center_index + (i - (num_methods - 1) / 2) * width
		hbar=plt.bar(x_index, data_mean[:, i], width=width, linewidth=2,
		        color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
		if data_std is not None:
		    plt.errorbar(x=x_index, y=data_mean[:, i],
		                 yerr=data_std[:, i], fmt='k.', elinewidth=3,capsize=4)
		if use_barlabel_x:
			if i in [2,3]:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+3,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_barlabe_y and i==2:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.08,data_mean[k,i]+1,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_downarrow:
			if i==1:
				for j in range(2,data_mean.shape[0]):
					ax.annotate(text='', xy=(x_index[j],data_mean[j,i]), xytext=(x_index[j],200), arrowprops=dict(arrowstyle='<->',lw=4))
					ax.text(x_index[j]-0.04, 160, '$\downarrow$'+f'{200-data_mean[j,i]:.0f}%', ha="center", va="center", rotation='vertical', size=labelsize ,fontweight='bold')
					# ax.text(center_index[j]-0.02,data_mean[j,i]+5,'$\downarrow$'+f'{200-data_mean[j,i]:.0f}%',fontsize = 16, fontweight='bold')
			else:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+5,f'{data_mean[k,i]:.2f}',fontsize = labelsize,fontweight='bold')

		if latency_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = data_mean[k,i]/data_mean[k,0]
					ax.text(xdx-0.3,data_mean[k,i]+2,f'{mult:.1f}\u00D7',fontsize = labelsize)
		if bandwidth_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = int(10**data_mean[k,i]/10**data_mean[k,0])
					ax.text(xdx-0.4,data_mean[k,i]+0.1,f'{mult}\u00D7',fontsize = labelsize)
		if latency_met_annot:
			if i>=1:
				for k,xdx in enumerate(x_index):
					mult = (-data_mean[k,i] + data_mean[k,0])/data_mean[k,0]*100
					if i==2:
						ax.text(xdx-0.07,data_mean[k,i],'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical',fontweight='bold')
					else:
						ax.text(xdx-0.07,data_mean[k,i],'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical')
		if breakdown_annot:
			if i == 2:
				ax.text(x_index[0]-0.07,data_mean[0,2]+0.015,'$\downarrow$'+f'{int(1000*(data_mean[0,0]-data_mean[0,2]))}ms',fontsize = lgsize,rotation='vertical',fontweight='bold')
				ax.text(x_index[2]-0.07,data_mean[2,2]+0.005,'$\u2191$'+f'{int(1000*(data_mean[2,2]))}ms',fontsize = lgsize,rotation='vertical')
		if bw_annot:
			if i>=1:
				for k,xdx in enumerate(x_index):
					mult = (-data_mean[k,i] + data_mean[k,0])/data_mean[k,0]*100
					ax.text(xdx-0.07,data_mean[k,i]+20,f'{data_mean[k,i]:.1f}'+'($\downarrow$'+f'{mult:.1f}%)',fontsize = lgsize,rotation='vertical')
		if showaccbelow:
			if i<=1:
				ax.text(2.3,-2.3, "Better", ha="center", va="center", rotation=90, size=labelsize,
				    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2))
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					if i!=1:
						ax.text(xdx-0.06,data_mean[k,i]-1.7,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize,rotation='vertical')
					else:
						ax.text(xdx-0.06,data_mean[k,i]-1.7,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize,rotation='vertical',fontweight='bold')
		if showrepaccbelow:
			if i<=1:
				ax.text(1.3,-7, "Better", ha="center", va="center", rotation=90, size=labelsize,
				    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2))
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					ax.text(xdx-0.06,data_mean[k,i]-3.2,'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical')
		if showcompbelow:
			if i<=1:
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					ax.text(xdx-0.06,data_mean[k,i]-8,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize-2,rotation='vertical')
	
	if c2s_annot:
		ax.text(2.8,33.2,"0.23dB-0.44dB loss\nby switching from\n3090 to 1080", ha="center", va="center", size=lgsize,fontweight='bold')
	if ncol>0:
		if legloc is None:
			plt.legend(bbox_to_anchor=bbox_to_anchor, fancybox=True,
			           loc='upper center', ncol=ncol, fontsize=lgsize, frameon=frameon)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=lgsize, frameon=frameon)
	fig.savefig(path, bbox_inches='tight')
	plt.close()

import scipy.interpolate

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def save_rate(R1, PSNR1, R2, PSNR2):
	lR1 = np.log(R1)
	lR2 = np.log(R2)

	# rate method
	p1 = np.polyfit(PSNR1, lR1, 3)
	p2 = np.polyfit(PSNR2, lR2, 3)

	# integration interval
	min_int = max(min(PSNR1), min(PSNR2))
	max_int = min(max(PSNR1), max(PSNR2))
	avg_int = (max_int + min_int)/2

	bw1,bw2 = np.exp(np.polyval(p1, avg_int)),np.exp(np.polyval(p2, avg_int))
	saving = (bw2-bw1)/bw2*100
	return int(saving),bw1,bw2

def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))
    avg_int = (max_int + min_int)/2
    print(np.exp(np.polyval(p1, avg_int)));print(np.exp(np.polyval(p2, avg_int)))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
	
    return avg_diff


def plot_RD_tradeoff():
	ncol = 1
	num_methods = 5
	bpps = [[[] for _ in range(num_methods)] for _ in range(5)]
	PSNRs = [[[] for _ in range(num_methods)] for _ in range(5)]
	SSIMs = [[[] for _ in range(num_methods)] for _ in range(5)]
	lSSIMs = [[[] for _ in range(num_methods)] for _ in range(5)]
	filenames = ['MCVC-IA-OLFT.cat.log','MCVC-Original.avg.log',
			  'x264-veryslow.avg.16.log','x265-veryslow.avg.16.log',]
			#   'x264-veryslow.avg.250.log','x265-veryslow.avg.250.log']
	pos_list = [0,2,3,4,5,6]
	for i,filename in enumerate(filenames):
		with open(filename, mode='r') as f:
			for l in f.readlines():
				l = l.split(',')
				if filename == 'MCVC-IA-OLFT.cat.log':
					cat,lvl,bpp,psnr0,ssim0,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6])
					r0,r_start,r_end = float(l[7]),float(l[8]),float(l[9])
					bpps[cat][1] += [bpp*1080*1920*views_of_category[cat]/1024/1024]
					PSNRs[cat][1] += [psnr0]
					SSIMs[cat][1] += [1-10**(-ssim0/10)]
					lSSIMs[cat][1] += [ssim0]
					bpp*=1.01
				else:
					cat,lvl,bpp,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
				if bpp > 0.2:
					continue
				bpp *= 1080*1920*views_of_category[cat]/1024/1024
				bpps[cat][pos_list[i]] += [bpp]
				PSNRs[cat][pos_list[i]] += [psnr]
				SSIMs[cat][pos_list[i]] += [1-10**(-ssim/10)]
				lSSIMs[cat][pos_list[i]] += [ssim]
	methods = ['Ours','Ours-OFF','SSF','x264','x265']#,'x264-UG','x265-UG']
	markersize = 16
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
	for cat in range(5):
		psnr_saving,psnr_r1,psnr_r2 = save_rate(bpps[cat][0], PSNRs[cat][0], bpps[cat][4], PSNRs[cat][4])
		ssim_saving,ssim_r1,ssim_r2 = save_rate(bpps[cat][0], lSSIMs[cat][0], bpps[cat][2], lSSIMs[cat][2])
		psnr_mean = sum(PSNRs[cat][0])/4
		ssim_mean = sum(SSIMs[cat][0])/4

		psnr1,psnr2 = sum(PSNRs[cat][0])/4, sum(PSNRs[cat][1])/4
		psnr_gain = psnr1 - psnr2
		ssim1,ssim2 = sum(SSIMs[cat][0])/4, sum(SSIMs[cat][1])/4
		ssim_gain = ssim1 - ssim2
		rate_mean = sum(bpps[cat][0])/4
		line_plot(bpps[cat],PSNRs[cat],methods,colors,
				f'images/psnr_{cat}.pdf',
				'Bandwidth Usage (Mbps)','PSNR (dB)',lbsize=24,lgsize=20,linewidth=2,
				ncol=ncol,markersize=markersize,bbox_to_anchor=None,xticks=[0,0.5,1,1.5,2],
				saving_annot=(psnr_saving,psnr_mean,psnr_r1,psnr_r2,rate_mean,psnr1,psnr2))
		
		line_plot(bpps[cat],SSIMs[cat],methods,colors,
				f'images/ssim_{cat}.pdf',
				'Bandwidth Usage (Mbps)','SSIM',lbsize=24,lgsize=20,linewidth=2,
				ncol=ncol,markersize=markersize,bbox_to_anchor=None,xticks=[0,0.5,1,1.5,2],
				saving_annot=[ssim_saving,ssim_mean,ssim_r1,ssim_r2,rate_mean,ssim1,ssim2])
		
def plot_bw():
	bpps = [[] for _ in range(5)]
	bw_save_list = [[] for _ in range(5)]
	bw_real_list = [[] for _ in range(5)]
	si_list = [[] for _ in range(5)]
	with open('MCVC-IA-OLFT.cat.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			cat,lvl,bpp,psnr0,ssim0,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6])
			r0,r_start,r_end = float(l[7]),float(l[8]),float(l[9])
			bpp *= 1080*1920*views_of_category[cat]/1024/1024
			bpps[cat] += [bpp]
			bw_save_list[cat] += [100-r_start/r0*100]
			bw_real_list[cat] += [r_end/r_start]
			si_list[cat] += [r_start]
				
	methods = ['Lobby','Retail','Office','Industry','Cafe']
	line_plot(bpps,bw_save_list,methods,colors,
			f'images/bw_save.pdf',
			'Bandwidth Usage (Mbps)','Bandwidth Saving (%)',lbsize=24,lgsize=15,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=(0.38,1.03),legloc=None,xlim=(0,.8),bw_annot=True)
	line_plot(bpps,si_list,methods,colors,
			f'images/si.pdf',
			'Bandwidth Usage (Mbps)','Sampling Interval',lbsize=24,lgsize=16,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='best',si_annot2=True)
	envs = [256,512,1024,2048]
	groupedbar(np.array(bw_real_list).T,None,'Actual BW Impact (%)', 
		'images/bw_impact.pdf',methods=methods,labelsize=24,xlabel='$\lambda$',
		envs=envs,ncol=3,width=1./7,sep=1,legloc='best',lgsize=16,bbox_to_anchor=(0.14,1),ylim=(0.995,1.008))
	
def plot_vary_compute():
	psnr_list = [[] for _ in range(4)]
	bpp_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.c2s.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			c2s,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			bpp *= 1080*1920*6/1024/1024
			bpp_list[lvl] += [bpp]
			psnr_list[lvl] += [psnr]
	methods = ['Base','1080', '2080', '3090']
	envs = [256,512,1024,2048]
	groupedbar(np.array(psnr_list),None,'PSNR (dB)', 
		'images/psnr_vs_c2s.pdf',methods=methods,labelsize=24,xlabel='$\lambda$',
		envs=envs,ncol=1,width=1./6,sep=1,legloc='best',lgsize=17,ylim=(27,34),c2s_annot=True)
	
def plot_nv():
	psnr_list = [[] for _ in range(4)]
	nv_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.nv.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			nv,lvl,bpp,psnr = int(l[0]),int(l[1]),float(l[2]),float(l[3]),
			psnr_list[lvl] += [psnr]
			nv_list[lvl] += [nv]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(nv_list,psnr_list,methods,colors,
			f'images/psnr_vs_nv.pdf',
			'# of Views','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='best',)
	
def plot_sr():
	psnr_list = [[] for _ in range(4)]
	bw_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.sr.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			sr,lvl,bpp,psnr,si = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[9])
			bpp *= 1080*1920*6/1024/1024
			psnr_list[lvl] += [psnr]
			bw_list[lvl] += [si]
	bw_list = np.array(bw_list)
	bw_list /= bw_list[:,-1:]
	bw_list = np.log10(bw_list)
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(bw_list,psnr_list,methods,colors,
			f'images/psnr_vs_sr.pdf',
			'Bandwidth Usage (%)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='best',
			xticks=[-2,-1,0],
			xticklabel=[1,10,100],sr_annot=True)
	
def plot_si():
	psnr_list = [[] for _ in range(4)]
	bw_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.si.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			si,lvl,psnr = float(l[0]),int(l[1]),float(l[3])
			psnr_list[lvl] += [psnr]
			si = 1/si
			bw_list[lvl] += [np.log10(si)]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(bw_list,psnr_list,methods,colors,
			f'images/psnr_vs_si.pdf',
			'Bandwidth Usage (%)','PSNR (dB)',lbsize=24,lgsize=16,linewidth=2,
			ncol=2,markersize=16,bbox_to_anchor=None,legloc='best',
			xticks = [-3,-2,-1,0],
			xticklabel= [0.1,1,10,100],si_annot=True)

	
def plot_mps():
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)] 
	with open('MCVC-IA-OLFT.mps.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			mps,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			x_list[lvl] += [mps*16]
			psnr_list[lvl] += [psnr]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_mps.pdf',
			'Cache Size (#Frames)','PSNR (dB)',lbsize=21,lgsize=16,linewidth=2,
			ncol=2,markersize=16,bbox_to_anchor=None,legloc='best',
			xticks=[0,100,200,300,400],
			xticklabel=[0,100,200,300,'Every'],
			mps_annot=True)

def plot_dr():
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)] 
	with open('MCVC-IA-OLFT.dr.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			dr,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			x_list[lvl] += [dr*113.456]
			psnr_list[lvl] += [psnr]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_dr.pdf',
			'#Streamed Frames (K)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=2,markersize=4,bbox_to_anchor=None,legloc='best',
			# ratio=0.46
			)
	
def plot_sisr():
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.sisr.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			sr,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			if sr == 0:
				sr = -4
			else:
				sr = np.log10(sr)
			x_list[lvl] += [sr]
			psnr_list[lvl] += [psnr]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_sisr.pdf',
			'Spatial Sampling Ratio','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='best',
			xticks = [-4,-3,-2,-1,0],
			xticklabel= [0,0.001,0.01,0.1,'1 (Ours)'],
			sisr_annot=True)
	
def plot_ablation():
	bpps = [[] for _ in range(4)]
	PSNRs = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.ablation.log', mode='r') as f:
		idx = 0
		for l in f.readlines():
			l = l.split(',')
			lvl,bpp,psnr = int(l[2]),float(l[3]),float(l[4])
			bpp *= 1080*1920*views_of_category[1]/1024/1024
			if idx%4 == 0:
				bpps[1] += [bpp]
				PSNRs[1] += [psnr]
			elif idx%4 == 2:
				bpps[2] += [bpp]
				PSNRs[2] += [psnr]
			idx += 1
			
	with open('MCVC-IA-OLFT.cat.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			cat,lvl,bpp,psnr0,psnr = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[5])
			bpp *= 1080*1920*views_of_category[1]/1024/1024
			if cat != 1:
				continue
			bpps[3] += [bpp]
			PSNRs[3] += [psnr0]
			bpps[0] += [bpp]
			PSNRs[0] += [psnr]
			
	methods = ['Default','w/o ACA','w/o SAR','w/o OL',]
	line_plot(bpps,PSNRs,methods,colors,
			f'images/ablation.pdf',
			'Bandwidth Usage (Mbps)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='lower right',ablation_annot=True)
	
def plot_longterm():
	# 1h per epoch/93759 frames, 5852 epochs
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)] 
	# with open('MCVC-IA-OLFT.cat.log', mode='r') as f:
	# 	for l in f.readlines():
	# 		l = l.split(',')
	# 		cat,lvl,bpp,psnr0,psnr = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[5])
	# 		if cat != 1:
	# 			continue
	# 		x_list[lvl] += [0]
	# 		psnr_list[lvl] += [psnr]
	with open('MCVC-IA-OLFT.longterm.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			lvl,epoch,bpp,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			x_list[lvl] += [(1+epoch)*5.852]
			psnr_list[lvl] += [psnr]
	methods = ['$\lambda=256$', '$\lambda=512$','$\lambda=1024$','$\lambda=2048$']
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_longterm.pdf',
			'#Trained Frames (K)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=None,legloc='best',xticks=[0,20,40,60],
			xticklabel=['0 (SAP)',20,40,60])

def plot_display():
	names = ['Apple Vision Pro', 'Meta\nQuest 3', 'Meta Quest Pro','Oculus\nQuest 2','Pico 4','Pimax Crystal QLED']
	pixels = [23,2064*2208*2/1e6,1800*1920*2/1e6,1832*1920*2/1e6,2160*2160*2/1e6,2880*2880*2/1e6]
	refresh = [90,120,90,120,90,160]
	release = ['June 4, 2023', 'May 31, 2023','October 27, 2021','September 15, 2020','September 21, 2022','May 30, 2022']
	xcoord = [3.5,3.4,1.8,0.7,2.8,2.4]
	markers = ['o' for _ in names] 
	markersize_list = [r**0.5*5 for r in refresh]

	x = [[xc] for xc in xcoord]
	y = [[pix] for pix in pixels]
	line_plot(x,y,names,colors,
			f'images/display_vs_year.pdf',
			'Release Year','#Pixels (M)',lbsize=24,lgsize=18,linewidth=2,
			ncol=0,markersize=16,bbox_to_anchor=None,legloc='best',
			xticks=[0,1,2,3,4],xticklabel=[2020,2021,2022,2023,2024],
			yticks=[5,10,15,20,23],
			markersize_list=markersize_list,markers=markers,
			display_annot=[(-2.2,-0.5),(-0.4,1.5),(0.2,-1),(-0.4,1.5),(-1,0),(-1,2),],
			)
	
def plot_camera():
	names = ['Wildtrack','SALSA','DukeMTMC','Assembly101',"Meta's 21-cam Rig","Google's 46-cam Rig","Meta's 16-cam Rig"]
	views  =[7,4,8,12,21,46,16]
	resolution = [1920*1080,1024*768,1920*1080,1920*1080,2028*2704,3840*2160,7680*4320]
	date = [2017,2016,2016,2022,2022,2020,2019]
	duration = [2,60,60,85,513]

	markers = ['o' for _ in names] 
	markersize_list = [r**0.5*0.01 for r in resolution]

	x = [[xc] for xc in date]
	y = [[v] for v in views]
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

	line_plot(x,y,names,colors,
			f'images/capture_vs_year.pdf',
			'Release Year','#Views',lbsize=24,lgsize=18,linewidth=2,
			ncol=0,markersize=16,bbox_to_anchor=None,legloc='best',
			markersize_list=markersize_list,markers=markers,
			display_annot=[(0.3,-1),(0.2,-3),(-0.2,3.5),(-2.5,-4.5),(-3,3),(-2.5,-2),(-2.5,2),],
			yticks=range(0,55,10),xticks=range(2016,2023,2)
			)

def plot_speed():
	y_list = []
	y_list += [[50]*6]
	dec_speed = [0.003514678156313797, 0.005066935391165316,
	0.003838256916031241, 0.005380996245269974,
	0.0041302489172667265, 0.005700193654124936,
	0.004484572874847799, 0.006108296576421708,
	0.004801990651059896, 0.006426506982650608,
	0.005413683934137225, 0.007108943206723779,]
	dec_speed = 1/np.array(dec_speed).reshape((6,2)).T
	y_list += [dec_speed[0]]
	y_list += [dec_speed[1]]
	x_list = [range(1,7) for _ in range(3)]
	methods = ['Capture', 'Server','Server (w/o ACE)']
	line_plot(x_list,y_list,methods,colors,
			f'images/speed_vs_views.pdf',
			'#Views','Processing Speed (fps)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=16,bbox_to_anchor=(0.73,0.08),legloc='best',
			# ratio=0.46
			)
	
def plot_longterm_nv():
	# 1h per epoch/93759 frames, 5852 epochs
	psnr_list = [[] for _ in range(6)]
	x_list = [[] for _ in range(6)]
	with open('MCVC-IA-OLFT.nv.1.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			nv,epoch,psnr = int(l[1]),int(l[2]),float(l[4]),
			x_list[nv-1] += [(1+epoch)*5.852]
			psnr_list[nv-1] += [psnr]
	methods = [1, 2,3,4,5,6,]
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_longterm_nv.pdf',
			'#Trained Frames (K)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=2,markersize=16,bbox_to_anchor=None,legloc='best',yticks=range(29,33),
			)

# larger marker, change naming, "selected" in SI, SR, annot compute
plot_speed()
exit(0)
plot_bw()
plot_mps()
plot_sisr()
plot_vary_compute()
plot_si()
plot_sr()
plot_camera()
plot_display()
plot_ablation()
plot_RD_tradeoff()
plot_dr()
plot_nv()
plot_longterm()
exit(0)
plot_longterm_nv()