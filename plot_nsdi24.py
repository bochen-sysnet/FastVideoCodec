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
				linestyles=None,use_text_arrow=False,fps_double_arrow=False,linewidth=None,markersize=None,
				bandlike=False,band_colors=None):
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
		elif bandlike:
			if linestyles is not None:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					linestyle = linestyles[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			else:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], 
					linewidth=linewidth, markersize=markersize)
			plt.fill_between(xx, yy - yerr[i], yy + yerr[i], color=band_colors[i], alpha=0.3)
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

def crate_array_of_empty_list(size):
	# Create the empty array
	data = np.empty(size, dtype=object)

	# Fill the array with empty lists
	data.fill([])
	for i in range(size[0]):
		for j in range(size[1]):
			data[i, j] = data[i, j].copy()
	return data

def plot_sp_cdf():
	bpp_records = [[],[]]
	psnr_records = [[],[]]
	methods = ['ELFVC','ELFVC-SP']
	for i in range(2):
		with open(f'../NSDI_logs/{methods[i]}.log','r') as f:
			line_count = 0
			for l in f.readlines():
				if line_count%2 == 0:
					l = l.split(',')
					lvl,bpp = int(l[0]),float(l[1])
				else:
					l = l[1:-2].split(',')
					l = np.char.strip(l)
					psnr_list = np.array(l).astype(float)
					psnr_records[i] += psnr_list.tolist()
					bpp_records[i] += [bpp] * len(psnr_list)
				line_count += 1

	cdf_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
	cdf_labels = ['w/o SP','w/ SP (Ours)']
	measurements_to_cdf(bpp_records,f'/home/bo/Dropbox/Research/NSDI24/images/sp_bpp_cdf.eps',cdf_labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.7,0.4),lfsize=16,ncol=1,lbsize=24,xlabel=f'BPP')
	measurements_to_cdf(psnr_records,f'/home/bo/Dropbox/Research/NSDI24/images/sp_psnr_cdf.eps',cdf_labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.24,1.02),lfsize=16,ncol=1,lbsize=24,xlabel=f'PSNR (dB)')

def plot_sp_vs_level():
	methods = ['ELFVC','ELFVC-SP']
	bpp_data = crate_array_of_empty_list((8, 2))
	psnr_data = crate_array_of_empty_list((8, 2))
	for i in range(2):
		with open(f'../NSDI_logs/{methods[i]}.log','r') as f:
			line_count = 0
			for l in f.readlines():
				if line_count%2 == 0:
					l = l.split(',')
					level,bpp = int(l[0]),float(l[1])
				else:
					l = l[1:-2].split(',')
					l = np.char.strip(l)
					psnr_list = np.array(l).astype(float)
					bpp_data[level,i] += [bpp]
					psnr_data[level,i] += [psnr_list.mean()]
				line_count += 1

	bar_colors = ['#1f77b4', '#ff7f0e']
	labels = ['w/o SP','w/ SP (Ours)']
	for data,ylabel,fname in zip([bpp_data,psnr_data],['BPP','PSNR (dB)'],['bpp','psnr']):
		# Calculate the mean of each list, handling empty lists as zero
		average = np.array([np.mean(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		std_dev = np.array([np.std(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		groupedbar(average,std_dev,ylabel, 
			f'/home/bo/Dropbox/Research/NSDI24/images/sp_{fname}_vs_level.eps',methods=labels,colors=bar_colors,ylim=((30,50) if fname=='psnr' else None),
			envs=[i for i in range(1,9)],ncol=1,sep=1,width=0.3,labelsize=24,lfsize=16,xlabel='Compression Level',legloc='best')

	

def plot_qerr():
	data = crate_array_of_empty_list((4, 8))
	with open(f'../NSDI_logs/ELFVC-SP.log','r') as f:
		line_count = 0
		for l in f.readlines():
			if line_count%2 == 0:
				l = l.split(',')
				lvl,flow_sp	,flow_q,res_sp,res_q = int(l[0]),float(l[4]),float(l[5]),float(l[6]),float(l[7])
				data[0,lvl].append(flow_sp)
				data[1,lvl].append(flow_q)
				data[2,lvl].append(res_sp)
				data[3,lvl].append(res_q)
			line_count += 1
	# Calculate the mean of each list, handling empty lists as zero
	average = np.array([np.mean(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
	std_dev = np.array([np.std(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)

	labels = ['Flow w/ SP','Flow w/o SP','Residual w/ SP','Residual w/o SP']
	colors_tmp = ['#4169E1', '#228B22', '#DC143C', '#9932CC']
	band_colors = ['#ADD8E6', '#98FB98', '#F08080', '#E6E6FA']


	groupedbar(average.T,std_dev.T,f'Quantization Jitter', 
		f'/home/bo/Dropbox/Research/NSDI24/images/qjitter_bar.eps',methods=labels,colors=colors_tmp,
		envs=[i for i in range(1,9)],ncol=1,sep=1,width=0.2,labelsize=24,lfsize=16,xlabel='Compression Level',legloc='best')

	# line_plot([range(1,9) for _ in range(4)],average,labels,colors_tmp,
	# 	'/home/bo/Dropbox/Research/NSDI24/images/qjitter_band.eps',
	# 	'Compression Level','Quantization Jitter',lbsize=24,lfsize=18,linewidth=2,markersize=4,
	# 	yerr=std_dev,band_colors=band_colors,bandlike=True,linestyles=linestyles,xticks=range(1,9))

def plot_se_per_video():
	datasets = ['UVG']#,'MCL-JCV']
	# same level for all, e.g., 0
	# one video per line
	bpp_data = []
	psnr_data = []
	epoch_data = []
	for dataset in datasets:
		with open(f'../NSDI_logs/ELFVC-SP.{dataset}.log','r') as f:
			line_count = 0
			for l in f.readlines():
				iteration = 0
				bpp_list = []; psnr_list = []; epoch_list = []
				for level,start,stage_name,_,bpp,psnr in eval(l):
					if stage_name != 'motion': break
					if level==0:
						bpp_list += [bpp]
						psnr_list += [psnr]
						epoch_list += [iteration]
						iteration += 1
				if level == 0:
					bpp_data += [bpp_list]
					psnr_data += [psnr_list]
					epoch_data += [epoch_list]
				line_count += 1

	line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

	band_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2']

	line_styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5)), (0, (1, 1))]

	labels = ['Bosphorus', 'ShakeNDry', 'Beauty', 'HoneyBee', 'ReadySetGo', 'YachtRide', 'Jockey']
	line_plot(epoch_data,bpp_data,labels,line_colors,
		'/home/bo/Dropbox/Research/NSDI24/images/bpp_evolution_by_video.eps',
		'# of Iterations','BPP',lbsize=24,lfsize=16,linewidth=2,markersize=4,ncol=1,
		linestyles=line_styles,xticks=range(0,35,5),yticks=[.05,.1,.15,.2,],bbox_to_anchor=(.55,1.02))
	line_plot(epoch_data,psnr_data,labels,line_colors,
		'/home/bo/Dropbox/Research/NSDI24/images/psnr_evolution_by_video.eps',
		'# of Iterations','PSNR (dB)',lbsize=24,lfsize=16,linewidth=2,markersize=4,ncol=1,
		linestyles=line_styles,xticks=range(0,35,5),yticks=[32,34,36,38],bbox_to_anchor=(.58,0.67))

def plot_se_vs_level():
	datasets = ['UVG','MCL-JCV']
	bpp_data = crate_array_of_empty_list((8, 2))
	psnr_data = crate_array_of_empty_list((8, 2))
	for dataset in datasets:
		with open(f'../NSDI_logs/ELFVC-SP.{dataset}.log','r') as f:
			for l in f.readlines():
				bpp_list = []; psnr_list = []
				for level,start,stage_name,_,bpp,psnr in eval(l):
					if stage_name != 'motion': break
					bpp_list += [bpp]; psnr_list += [psnr]
				bpp_data[level,0].append(1-bpp_list[1]/bpp_list[0])
				bpp_data[level,1].append(1-bpp_list[-1]/bpp_list[0])
				psnr_data[level,0].append(psnr_list[1] - psnr_list[0])
				psnr_data[level,1].append(psnr_list[-1] - psnr_list[0])

	bar_colors = ['#1f77b4', '#ff7f0e']
	labels = ['One Iter (Ours)','Converged']
	for data,ylabel,fname in zip([bpp_data,psnr_data],['BPP Reduction (%)','Improved PSNR (dB)'],['bpp','psnr']):
		# Calculate the mean of each list, handling empty lists as zero
		average = np.array([np.mean(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		std_dev = np.array([np.std(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		groupedbar(average,std_dev,ylabel, 
			f'/home/bo/Dropbox/Research/NSDI24/images/se_{fname}_improvement_vs_level.eps',methods=labels,colors=bar_colors,
			envs=[i for i in range(1,9)],ncol=1,sep=1,width=0.3,labelsize=24,lfsize=16,xlabel='Compression Level',legloc='best')

def plot_se_cdf():
	datasets = ['UVG','MCL-JCV']
	bpp_records = [[],[],[]]
	psnr_records = [[],[],[]]
	iteration_records = [[],[]]
	for dataset in datasets:
		with open(f'../NSDI_logs/ELFVC-SP.{dataset}.log','r') as f:
			for l in f.readlines():
				bpp_list = []; psnr_list = []
				for level,start,stage_name,_,bpp,psnr in eval(l):
					if stage_name != 'motion': break
					bpp_list += [bpp]; psnr_list += [psnr]
				bpp_records[0].append(bpp_list[0])
				bpp_records[1].append(bpp_list[1])
				bpp_records[2].append(bpp_list[-1])
				psnr_records[0].append(psnr_list[0])
				psnr_records[1].append(psnr_list[1])
				psnr_records[2].append(psnr_list[-1])
				iteration_records[0].append(1)
				iteration_records[1].append(len(bpp_list)-1)

	cdf_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
	cdf_labels = ['w/o SE','One Iter (Ours)','Converged']
	measurements_to_cdf(bpp_records,f'/home/bo/Dropbox/Research/NSDI24/images/se_bpp_cdf.eps',cdf_labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.7,0.4),lfsize=16,ncol=1,lbsize=24,xlabel=f'BPP')
	measurements_to_cdf(psnr_records,f'/home/bo/Dropbox/Research/NSDI24/images/se_psnr_cdf.eps',cdf_labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.24,1.02),lfsize=16,ncol=1,lbsize=24,xlabel=f'PSNR (dB)')
	measurements_to_cdf(iteration_records,f'/home/bo/Dropbox/Research/NSDI24/images/se_iter_cdf.eps',['Ours','For Convergence'],linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.4,1.02),lfsize=16,ncol=1,lbsize=24,xlabel=f'# of Iterations',xticks=[0,1,10,20,30])


def plot_RD_tradeoff():
	SPSNRs = [[32.20030428647995, 34.520895831346515, 35.93721253871918, 37.8684334590435, 39.51266849374771, 40.823672102928164, 42.145868400096894, 43.44012289690971], [33.167340933322905, 34.58496722102165, 36.156650943040844, 37.54983545994759, 38.80151827955246, 40.20092285299301, 41.51993577432633, 42.80891443252563], [31.911170615196227, 33.59662669181824, 35.073301779270174, 36.3265318365097, 37.370484233379365, 38.193767918348314, 38.84856410169601, 39.35588079452515], [33.23122785019875, 34.62406644463539, 35.83494013476372, 36.84865824365616, 37.70339743518829, 38.445705357074736, 39.0738078956604, 39.51968069934845], [32.95162628340721, 34.39114833903313, 35.67984473323822, 36.77041100358963, 37.67305509305, 38.44268340206146, 39.0731332988739, 39.53226850509643], [33.075983564376834, 34.45408898758888, 35.66986226463318, 36.71133078813553, 37.59897288942337, 38.343172566890715, 38.947341213226316, 39.40737556648254], [32.94724491167069, 34.29442249751091, 35.48730764818192, 36.516958021640775, 37.373835401058194, 38.09869503426552, 38.70549611997605, 39.18615546035767], [33.06004134774208, 34.43566031122208, 35.63771190547943, 36.680811815023425, 37.53766157031059, 38.263620466709135, 38.868735751628876, 39.33174180984497]]
	Sbpps = [[0.025, 0.06799999999999999, 0.1, 0.18000000000000005, 0.28, 0.38, 0.5399999999999999, 0.7499999999999998], [0.05218125, 0.081607375, 0.12142825, 0.1812308125, 0.2590743125, 0.36626200000000003, 0.5203261874999999, 0.7336966875], [0.02382375, 0.03850100000000001, 0.06485250000000001, 0.11969093749999998, 0.23905893749999996, 0.47534131250000006, 0.8754845, 1.4809526874999999], [0.026881250000000002, 0.042514937499999995, 0.06942899999999999, 0.120578, 0.2225614375, 0.43596099999999993, 0.8270014375000001, 1.4458957499999998], [0.025379250000000003, 0.039885, 0.0653433125, 0.11526875000000002, 0.215117875, 0.42379268750000004, 0.8072253125000001, 1.4425223125], [0.020962124999999998, 0.035693562500000005, 0.06224675000000002, 0.115982, 0.22779037500000007, 0.453151875, 0.8685715, 1.5292138125], [0.016587437500000003, 0.027825500000000003, 0.04661843749999999, 0.08143237499999999, 0.149085375, 0.2852063125, 0.5427845, 0.98032825], [0.0165040625, 0.028121999999999994, 0.04716850000000001, 0.0833735, 0.15286724999999998, 0.300959875, 0.5819153125, 1.0572071250000001]]

	SPSNRs = np.array(SPSNRs)
	Sbpps = np.array(Sbpps)

	sc_labels = ['Vesper','ELFVC','SSF','x264-veryfast','x264-medium','x264-veryslow','x265-veryfast','x265-medium','x265-veryslow',]

	colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
	line_plot(Sbpps,SPSNRs,sc_labels,colors_tmp,
			'/home/bo/Dropbox/Research/NSDI24/images/rdtradeoff.eps',
			'Bit Per Pixel','PSNR (dB)',yticks=range(32,44),lbsize=24,lfsize=14,linewidth=1,markersize=4)

def plot_QoE_cdf_breakdown():
	# 16543747 bps=15.8Mbps
	# 4074145 mbps=3.9Mbps
	labels_tmp = ['Vesper','ELFVC','SSF','x264f','x264m','x264s','x265f','x265m','x265s']
	colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
	metric_list = []
	hw = 3090
	for k,metric in enumerate(['QoE','quality','rebuffer','bw']):
		names = ['QoE','Quality','Rebuffer Rate','BW Consumption']
		meanQoE_all = [];stdQoE_all = []
		for trace in range(2):
			datafile = f'/home/bo/Dropbox/Research/NSDI24/data/{metric}_{trace}_{hw}_1000.data'
			with open(datafile,'r') as f:
				line = f.readlines()[0]
			QoE_matrix = eval(line)
			QoE_matrix = np.array(QoE_matrix)
			QoE_min,QoE_max = QoE_matrix.min(),QoE_matrix.max()
			QoE_matrix = (QoE_matrix - QoE_min) / (QoE_max - QoE_min) 
			measurements_to_cdf(QoE_matrix,f'/home/bo/Dropbox/Research/NSDI24/data/{metric}cdf_{trace}_{hw}.eps',labels_tmp,linestyles=linestyles,
				colors=colors_tmp,bbox_to_anchor=(.14,1.02),lfsize=16,ncol=1,lbsize=24,xlabel=f'Normalized {names[k]}')
			meanQoE = QoE_matrix.mean(axis=1)
			stdQoE = QoE_matrix.std(axis=1)
			meanQoE_all += [meanQoE]
			stdQoE_all += [stdQoE]
		meanQoE_all = np.stack(meanQoE_all).reshape(2,-1)
		# print(meanQoE_all.tolist())
		stdQoE_all = np.stack(stdQoE_all).reshape(2,-1)
		if k == 0:
			ncol = 1
			labelsize=18
		else:
			labelsize=24
			ncol = 0
		groupedbar(meanQoE_all,stdQoE_all,f'Normalized {names[k]}', 
			f'/home/bo/Dropbox/Research/NSDI24/data/{metric}mean.eps',methods=labels_tmp,colors=colors_tmp,
			envs=['Limited BW','Adequate BW'],ncol=ncol,sep=1,width=0.1,labelsize=labelsize,lfsize=16,bbox_to_anchor=(1.22,1.05),xlabel='',ratio=.7)
		if metric == 'QoE':
			for line in meanQoE_all.tolist():
				ours,t_top1,l_top1 = line[0],max(line[1:3]),max(line[3:])
				m1,m2=(ours - t_top1)/t_top1,(ours - l_top1)/l_top1
				metric_list += [[m1,m2]]
	metric_list = np.array(metric_list)
	print(metric_list)
	print(metric_list.mean(axis=0))
	print(metric_list[:3].mean(axis=0))
	print(metric_list[3:].mean(axis=0))


def plot_QoE_ablation():
	# 16543747 bps=15.8Mbps
	# 4074145 mbps=3.9Mbps
	labels_tmp = ['Base','Base+SP','Base+SP+SE']
	colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
	hw = 3090
	for k,metric in enumerate(['QoE','quality','rebuffer','bw']):
		names = ['QoE','Quality','Rebuffer Rate','BW Consumption']
		meanQoE_all = [];stdQoE_all = []
		for trace in range(2):
			datafile = f'/home/bo/Dropbox/Research/NSDI24/ablation/{metric}_{trace}_{hw}_1000.data'
			with open(datafile,'r') as f:
				line = f.readlines()[0]
			QoE_matrix = eval(line)
			QoE_matrix = np.array(QoE_matrix)
			QoE_min,QoE_max = QoE_matrix.min(),QoE_matrix.max()
			QoE_matrix = (QoE_matrix - QoE_min) / (QoE_max - QoE_min) 
			measurements_to_cdf(QoE_matrix,f'/home/bo/Dropbox/Research/NSDI24/ablation/{metric}cdf_{trace}_{hw}.eps',labels_tmp,linestyles=linestyles,
				colors=colors_tmp,bbox_to_anchor=(.14,1.02),lfsize=16,ncol=1,lbsize=24,xlabel=f'Normalized {names[k]}')
			meanQoE = QoE_matrix.mean(axis=1)
			stdQoE = QoE_matrix.std(axis=1)
			meanQoE_all += [meanQoE]
			stdQoE_all += [stdQoE]
		meanQoE_all = np.stack(meanQoE_all).reshape(2,-1)
		stdQoE_all = np.stack(stdQoE_all).reshape(2,-1)
		if k == 0:
			ncol = 1
			labelsize=18
		else:
			labelsize=24
			ncol = 0
		groupedbar(meanQoE_all,stdQoE_all,f'Normalized {names[k]}', 
			f'/home/bo/Dropbox/Research/NSDI24/ablation/{metric}mean.eps',methods=labels_tmp,colors=colors_tmp,
			envs=['Limited BW','Adequate BW'],ncol=ncol,sep=1,width=0.1,labelsize=labelsize,lfsize=16,bbox_to_anchor=(1.22,1.05),xlabel='',ratio=.7)


plot_se_per_video()
exit(0)
plot_sp_vs_level()

plot_sp_cdf()

plot_se_cdf()

plot_se_vs_level()

plot_qerr()
exit(0)

# Overall RD tradeoff
plot_RD_tradeoff()

##############################Overall QoE and breakdown#############################
plot_QoE_cdf_breakdown()