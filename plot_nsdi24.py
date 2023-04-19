#!/usr/bin/python

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
lfsize = 18
labelsize = 24
labelsize_s,labelsize_b = 24,32
linewidth = 4
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
# colors = ['#D00C0E','#E09C1A','#08A720','#86A8E7','#9D5FFB','#D65780']
labels = ['ELVC','H.264','H.265','DVC','RLVC']
markers = ['o','P','s','D','>','^','<','v','*']
hatches = ['/' ,'\\','--','x', '+', 'O','-','o','.','*']
linestyles = ['solid','dotted','dashed','dashdot', (0, (3, 5, 1, 5, 1, 5))]
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


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

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

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b-8,legloc='best',
				xticks=None,yticks=None,xticklabel=None,ncol=None, yerr=None,markers=markers,
				use_arrow=False,arrow_coord=(0.4,30),ratio=None,bbox_to_anchor=(1.1,1.2),use_doublearrow=False,
				linestyles=None,use_text_arrow=False,fps_double_arrow=False,linewidth=None,markersize=None):
	if linewidth is None:
		linewidth = 2
	if markersize is None:
		markersize = 8
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			if linestyles is not None:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					linestyle = linestyles[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			else:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linewidth=linewidth, markersize=markersize)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		if xticklabel is None:
			plt.xticks(xticks,fontsize=lfsize)
		else:
			plt.xticks(xticks,xticklabel,fontsize=lfsize)
	ax.tick_params(axis='both', which='major', labelsize=lbsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lbsize)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	if use_doublearrow:
		plt.axhline(y = YY[0,0], color = color[0], linestyle = '--')
		ax.annotate(text='', xy=(2,YY[0,0]), xytext=(2,YY[0,1]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[0]))
		ax.text(
		    2.5, 25, "76% less time", ha="center", va="center", rotation='vertical', size=lfsize, color = color[0])
		plt.axhline(y = YY[2,0], color = color[2], linestyle = '--')
		ax.annotate(text='', xy=(6,YY[2,0]), xytext=(6,YY[2,5]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[2]))
		ax.text(
		    6.5, 23, "87% less time", ha="center", va="center", rotation='vertical', size=lfsize,color = color[2])
	if fps_double_arrow:
		for i in range(3):
			ax.annotate(text='', xy=(31+i*0.5,YY[3*i,0]), xytext=(31+i*0.5,YY[0+3*i,-1]), arrowprops=dict(arrowstyle='<->',lw=2, color = color[i*3]))
			ax.text(
			    32+i*0.5, (YY[3*i,-1]+YY[i*3,0])/2+i*0.5, f"{YY[3*i,-1]/YY[3*i,0]:.1f}X", ha="center", va="center", rotation='vertical', size=lfsize, color = color[i*3])
	if use_text_arrow:
		ax.annotate('Better speed and\ncoding efficiency trade-off', xy=(XX[2][-1]+1, YY[2,-1]+20),  xycoords='data',
            xytext=(0.25, 0.4), textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->',lw=2),size=lbsize,
            # horizontalalignment='right', verticalalignment='top'
            )

	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lfsize)
		else:
			plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	# plt.xlim((0.8,3.2))
	# plt.ylim((-40,90))
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

def bar_plot(avg,std,label,path,color,ylabel,labelsize=24,yticks=None):
	N = len(avg)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	ax.set_axisbelow(True)
	if std is not None:
		hbar = ax.bar(ind, avg, width, color=color, \
			yerr=std, error_kw=dict(lw=1, capsize=1, capthick=1))
	else:
		hbar = ax.bar(ind, avg, width, color=color, \
			error_kw=dict(lw=1, capsize=1, capthick=1))
	ax.set_ylabel(ylabel, fontsize = labelsize)
	ax.set_xticks(ind,fontsize=labelsize)
	ax.set_xticklabels(label, fontsize = labelsize)
	ax.bar_label(hbar, fmt='%.2f', fontsize = labelsize,fontweight='bold',padding=8)
	if yticks is not None:
		plt.yticks( yticks,fontsize=18 )
	# xleft, xright = ax.get_xlim()
	# ybottom, ytop = ax.get_ylim()
	# ratio = 0.3
	# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

def hbar_plot(avg,std,label,path,color,xlabel):
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

	d = .03 # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='r', clip_on=False)
	ax1.plot((1-d,1+d), (-d,+d), **kwargs)
	ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d,+d), (1-d,1+d), **kwargs)
	ax2.plot((-d,+d), (-d,+d), **kwargs)

	ax1.bar_label(hbars1, fmt='%.2f', fontsize = labelsize_b-8)
	ax2.bar_label(hbars2, fmt='%.2f', fontsize = labelsize_b-8)
	ax1.set_yticks(y_pos, labels=label, fontsize = labelsize_b)
	ax1.invert_yaxis()  

	ax1.set_xticks([])
	ax2.set_xticks([])

	plt.tight_layout()
	fig.text(0.55, 0, xlabel, ha='center', fontsize = labelsize_b-8)
	fig.savefig(path,bbox_inches='tight')


def measurements_to_cdf(latency,epsfile,labels,xticks=None,xticklabel=None,linestyles=linestyles,colors=colors,
                        xlabel='Normalized QoE',ylabel='CDF',ratio=None,lbsize = 18,lfsize = 18,linewidth=4,bbox_to_anchor=(0.5,-.5),
                        loc='upper center',ncol=3):
    # plot cdf
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    for i,latency_list in enumerate(latency):
        N = len(latency_list)
        cdf_x = np.sort(np.array(latency_list))
        cdf_p = np.array(range(N))/float(N)
        plt.plot(cdf_x, cdf_p, color = colors[i], label = labels[i], linewidth=linewidth, linestyle=linestyles[i])
    plt.xlabel(xlabel, fontsize = lbsize)
    plt.ylabel(ylabel, fontsize = lbsize)
    if xticks is not None:
        plt.xticks(xticks,fontsize=lbsize)
    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if ratio is not None:
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    if bbox_to_anchor is not None:
    	plt.legend(loc=loc,fontsize = lfsize,bbox_to_anchor=bbox_to_anchor, fancybox=True,ncol=ncol)
    else:
    	plt.legend(loc=loc,fontsize = lfsize, fancybox=True,ncol=ncol)
    plt.tight_layout()
    fig.savefig(epsfile,bbox_inches='tight')
    plt.close()

def groupedbar(data_mean,data_std,ylabel,path,yticks=None,envs = [2,3,4],colors=colors,
				methods=['Ours','Standalone','Optimal','Ours*','Standalone*','Optimal*'],use_barlabel_x=False,use_barlabe_y=False,
				ncol=3,bbox_to_anchor=(0.46, 1.28),sep=1.,width=0.5,xlabel=None,legloc=None,labelsize=labelsize_b,ylim=None,lfsize=labelsize_b,
				rotation=None,bar_label_dxdy=(-0.3,5),use_realtime_line=False,additional_y=None,ratio=None):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)*sep
	# colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']

	ax.grid()
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['top'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	if additional_y is not None:
		xtick_loc = center_index.tolist() + [4.5]
		envs += ['CPU']
	else:
		xtick_loc = center_index

	if rotation is None:
		plt.xticks(xtick_loc, envs, size=labelsize)
	else:
		plt.xticks(xtick_loc, envs, size=labelsize, rotation=rotation)
	plt.yticks(fontsize=labelsize)
	ax.set_ylabel(ylabel, size=labelsize)
	if xlabel is not None:
		ax.set_xlabel(xlabel, size=labelsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=labelsize)
	if ylim is not None:
		ax.set_ylim(ylim)
	for i in range(num_methods):
		x_index = center_index + (i - (num_methods - 1) / 2) * width
		hbar=plt.bar(x_index, data_mean[:, i], width=width, linewidth=2,
		        color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
		if data_std is not None:
		    plt.errorbar(x=x_index, y=data_mean[:, i],
		                 yerr=data_std[:, i], fmt='k.', elinewidth=2,capsize=4)
		if use_barlabel_x:
			for k,xdx in enumerate(x_index):
				if data_mean[k,i]>1:
					ax.text(xdx+bar_label_dxdy[0],data_mean[k,i]+bar_label_dxdy[1],f'{data_mean[k,i]:.1f}',fontsize = labelsize, fontweight='bold')
				else:
					ax.text(xdx+bar_label_dxdy[0],data_mean[k,i]+bar_label_dxdy[1],f'{data_mean[k,i]:.2f}',fontsize = labelsize, fontweight='bold')
		if use_barlabe_y and i==1:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.02,data_mean[k,i]+.02,f'{data_mean[k,i]:.4f}',fontsize = 18, rotation='vertical',fontweight='bold')
	if additional_y is not None:
		for i in range(additional_y.shape[0]):
			x_index = 4.5 + (i - (additional_y.shape[0] - 1) / 2) * width
			hbar=plt.bar(x_index, additional_y[i], width=width, linewidth=2,
		        color=colors[i+num_methods], label=methods[i+num_methods], hatch=hatches[i+num_methods], edgecolor='k')

	if use_realtime_line:
		plt.axhline(y = 30, color = '#DB1F48', linestyle = '--')
		# ax.text(7, 48, "4.5X more likely", ha="center", va="center", rotation='vertical', size=lbsize,fontweight='bold')
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	if ncol>0:
		if legloc is None:
			plt.legend(bbox_to_anchor=bbox_to_anchor, fancybox=True,
			           loc='upper center', ncol=ncol, fontsize=lfsize)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=lfsize)
	plt.tight_layout()
	fig.savefig(path, bbox_inches='tight')
	plt.close()

def plot_clustered_stacked(dfall, filename, labels=None, horizontal=False, xlabel='', ylabel='',**kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
    fig = plt.figure()
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=['#DB1F48','#1C4670',],
                      edgecolor='k',
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(hatches[i//n_col]) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.tick_params(axis='both', which='major', labelsize=20)
    axe.set_xlabel(xlabel, size=24)
    axe.set_ylabel(ylabel, size=24)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="white", hatch=hatches[i],edgecolor='black'))

    n2 = []
    for i,clr in enumerate(['#DB1F48','#1C4670',]):
    	n2.append(axe.bar(0, 0, color=clr))

    if labels is not None:
        if not horizontal:
            # l1 = axe.legend(h[:n_col], l[:n_col], loc=[.01, 0.78], fontsize=18)
            l3 = plt.legend(n2, ['Motion','Residual'], loc=[.01, 0.78], fontsize=18) 
            l2 = plt.legend(n, labels, loc=[.01, 0.47], fontsize=18) 
        else:
            # l1 = axe.legend(h[:n_col], l[:n_col], loc=[.68, 0.78], fontsize=18)
            l3 = axe.legend(n2, ['Enc','Dec'], loc=[.68, 0.78], fontsize=18) 
            l2 = plt.legend(n, labels, loc=[.68, 0.47], fontsize=18) 
    axe.add_artist(l3)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    return axe

# Sbpps = [
# [0.0211,0.0339,0.0563,0.0967,0.1705,0.3058,0.5590,1.0080],
# [0.0231,0.0361,0.0583,0.0958,0.1591,0.2681,0.4726,0.8669],
# [0.0216,0.0334,0.0538,0.0889,0.1482,0.2515,0.4520,0.8534],
# [0.0189,0.0323,0.0559,0.0975,0.1706,0.3075,0.5882,1.1209],
# [0.0138,0.0227,0.0376,0.0628,0.1055,0.1800,0.3172,0.5761],
# [0.0136,0.0229,0.0382,0.0643,0.1086,0.1852,0.3282,0.6105],
# ]
# SPSNRs = [
# [32.72,34.47,36.00,37.28,38.29,39.00,39.52,39.91],
# [34.02,35.46,36.72,37.76,38.59,39.20,39.67,40.00],
# [33.75,35.23,36.56,37.69,38.57,39.21,39.68,40.02],
# [34.01,35.41,36.65,37.69,38.52,39.14,39.59,39.94],
# [33.71,35.08,36.30,37.37,38.24,38.92,39.43,39.78],
# [33.82,35.22,36.46,37.54,38.42,39.09,39.56,39.89],
# ]
# sc_labels = ['x264-veryfast','x264-medium','x264-veryslow','x265-veryfast','x265-medium','x265-veryslow',]
# line_plot(Sbpps,SPSNRs,sc_labels,colors,
# 		'/home/bo/Dropbox/Research/NSDI24/images/rdtradeoff_2k.eps',
# 		'Bit Per Pixel','PSNR (dB)',lbsize=24,lfsize=18)


SPSNRs = [
[29.91,31.6,33.17,34.45,35.43,36.13,36.58],
[29.22,31.01,32.33,33.16,33.81,34.48,34.95,35.17],
[29.13,31,32.34,33.23,33.93,34.23,34.89,35.59],
[28.69,30.19,31.46,32.60,33.71,35.00,36.17],
[28.696,30.125,31.45,32.64,33.24,35.58,36.56,37.52],
[28.74]
]
# 5,3,2; 5,5,5
Sbpps = [
[0.0667,0.1099,0.1784,0.2847,0.4486,0.706,1.1164],
[0.0792,0.1280,0.1926,0.2776,0.3364,0.4715,0.6509,0.8680],
[0.0634,0.1098,0.1684,0.2439,0.3399,0.4767,0.6794,0.9825],
[0.0730,0.1238,0.1965,0.2991,0.4348,0.6325,0.8969],
[0.070,0.1130,0.1988,0.31,0.3982,0.7461,1.17,1.51],
[0.071]
]
sc_labels = ['x265','DVC','RLVC','SSF','ELFVC','Vesper']
line_plot(Sbpps,SPSNRs,sc_labels,colors,
		'/home/bo/Dropbox/Research/NSDI24/images/rdtradeoff_256.eps',
		'Bit Per Pixel','PSNR (dB)',lbsize=24,lfsize=18)
