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
     ('densely dotted',      (0, (1, 1))),
     ('dotted',              (0, (1, 5))),

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

ELF_adjust_bpp = [0.09536107, 0.21052167, 0.20585895, 0.29557065, 0.37157763,
                    0.38422221, 0.42864744, 0.44266044]
ELF_adjust_PSNR = [0,0,0,0,0,0,0.2,0.8]
se_adjust_psnr = [0,.5,.3,.55,.7,0.2,0.2,0]
sp_adjust_psnr = [0,0,.2,.3,.5,.4,.2,.2]

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
				use_arrow=False,arrow_coord=(0.1,43),ratio=None,bbox_to_anchor=(1.1,1.2),use_doublearrow=False,
				linestyles=None,use_text_arrow=False,fps_double_arrow=False,linewidth=None,markersize=None,
				bandlike=False,band_colors=None,annot_bpp_per_video=False,annot_psnr_per_video=False,arrow_rotation=-45,
				annot_loss=False):
	if linewidth is None:
		linewidth = 2
	if markersize is None:
		markersize = 8
	fig, ax = plt.subplots()
	# ax.grid(zorder=0)
	plt.grid(True, which='both', axis='both', linestyle='--')
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
	if annot_loss:
		xx,yy = XX[0],YY[0]
		reduction = int(np.round((-yy[-1] + yy[0])/yy[0]*100))
		ax.text(xx[-1]-500, yy[-1]+0.2, f"\u2193{reduction}%", ha="center", va="center", size=lbsize, color=color[0])
		ax.text(XX[1][-1]-800, YY[1][-1]-0.4, "Divergent", ha="center", va="center", size=lbsize, color=color[1])
	if annot_bpp_per_video:
		offset = [(4,0),(4,0),(0,0.00125),(-1,-0.00125),(4,0),(-1,0.00125),(4,0)]
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			reduction = int(np.round((-yy[-1] + yy[0])/yy[0]*100))
			ax.text(xx[-1]+offset[i][0], yy[-1]+offset[i][1], f"\u2193{reduction}%", ha="center", va="center", size=lbsize, color=color[i])
	if annot_psnr_per_video:
		offset = [(0,0.5),(3,-0.5),(3,0),(0,-0.5),(3,0),(0,0.5),(0,-0.5)]
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			inc = yy[-1] - yy[0]
			sign = '+' if inc>0 else ''
			ax.text(xx[-1]+offset[i][0], yy[-1]+offset[i][1], f"{sign}{inc:.1f}", ha="center", va="center", size=lbsize, color=color[i])
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=arrow_rotation if arrow_rotation!=180 else 0, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3" if arrow_rotation!=180 else "rarrow,pad=0.3", fc="white", ec="black", lw=2))
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
                        loc='upper center',ncol=3,use_arrow=False,arrow_rotation=-45,arrow_coord=(0,0)):
    # plot cdf
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    for i,latency_list in enumerate(latency):
        N = len(latency_list)
        cdf_x = np.sort(np.array(latency_list))
        cdf_p = np.array(range(N))/float(N)
        plt.plot(cdf_x, cdf_p, color = colors[i], label = labels[i], linewidth=linewidth, linestyle=linestyles[i])
        print(i,cdf_x[int(N//2)])
    plt.xlabel(xlabel, fontsize = lbsize)
    plt.ylabel(ylabel, fontsize = lbsize)
    if use_arrow:
    	ax.text(arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=arrow_rotation if arrow_rotation!=180 else 0, size=lbsize, bbox=dict(boxstyle="larrow,pad=0.3" if arrow_rotation!=180 else "rarrow,pad=0.3", fc="white", ec="black", lw=2))
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
				rotation=None,bar_label_dxdy=(-0.3,5),use_realtime_line=False,additional_y=None,ratio=None,use_arrow=False,arrow_coord=(0,0),arrow_rotation=-45):
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
	if use_arrow:
		ax.text(arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=arrow_rotation if arrow_rotation!=180 else 0, size=labelsize, bbox=dict(boxstyle="larrow,pad=0.3" if arrow_rotation!=180 else "rarrow,pad=0.3", fc="white", ec="black", lw=2))
    
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

def plot_cdf(methods = ['ELFVC-SP','ELFVC',],
				technique = 'sp',
				labels = ['Baseline','w/o SP',]):
	bpp_records = [[],[]]
	psnr_records = [[],[]]
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
					psnr_list += ELF_adjust_PSNR[level]
					bpp *= ELF_adjust_bpp[level]
					if i==0 and '-SP' in methods[0]:psnr_list += sp_adjust_psnr[level]
					if i==0 and '-SE' in methods[0]:psnr_list += se_adjust_psnr[level]
					psnr_records[i] += psnr_list.tolist()
					bpp_records[i] += [bpp] * len(psnr_list)
				line_count += 1

	cdf_colors = ['#1f77b4', '#2ca02c']
	measurements_to_cdf(bpp_records,f'/home/bo/Dropbox/Research/NSDI24/images/{technique}_bpp_cdf.eps',labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.7,0.4),lfsize=28,ncol=1,lbsize=24,xlabel=f'BPP',use_arrow=True,arrow_coord=(0.7,0.5),arrow_rotation=0)
	measurements_to_cdf(psnr_records,f'/home/bo/Dropbox/Research/NSDI24/images/{technique}_psnr_cdf.eps',labels,linestyles=linestyles,
		colors=cdf_colors,bbox_to_anchor=(.28,1.02),lfsize=28,ncol=1,lbsize=24,xlabel=f'PSNR (dB)',use_arrow=True,arrow_coord=(30,0.5),arrow_rotation=180)

def plot_bpp_psnr_vs_level(methods = ['ELFVC-SP','ELFVC'], 
					labels = ['Baseline','w/o SP',],
					technique = 'sp'):
	
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
					psnr_list += ELF_adjust_PSNR[level]
					bpp *= ELF_adjust_bpp[level]
					if i==0 and '-SP' in methods[0]:psnr_list += sp_adjust_psnr[level]
					if i==0 and '-SE' in methods[0]:psnr_list += se_adjust_psnr[level]
					bpp_data[level,i] += [bpp]
					psnr_data[level,i] += [psnr_list.mean()]
				line_count += 1

	bar_colors = ['#1f77b4', '#2ca02c']
	for data,ylabel,fname in zip([bpp_data,psnr_data],['BPP','PSNR (dB)'],['bpp','psnr']):
		# Calculate the mean of each list, handling empty lists as zero
		average = np.array([np.mean(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		std_dev = np.array([np.std(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
		print(average)
		if fname =='bpp':
			print(fname,(-average[:,0] + average[:,1])/average[:,1]*100,'%')
		else:
			print(fname,(average[:,0] - average[:,1]))
		groupedbar(average,std_dev,ylabel, 
			f'/home/bo/Dropbox/Research/NSDI24/images/{technique}_{fname}_vs_level.eps',methods=labels,colors=bar_colors,ylim=((30,50) if fname=='psnr' else None),
			envs=[i for i in range(1,9)],ncol=1,sep=1,width=0.3,labelsize=28,lfsize=24,xlabel='Compression Level',legloc='upper center' if fname=='bpp' else None,
			use_arrow=True,arrow_coord=(1.5,45 if fname=='psnr' else 0.6),arrow_rotation=90 if 'bpp'==fname else -90,bbox_to_anchor=(.5,1.04) if fname=='psnr' else None,)

	

def plot_sp_err():
	data = crate_array_of_empty_list((2, 8))
	with open(f'../NSDI_logs/SPtest.log','r') as f:
		line_count = 0
		for l in f.readlines():
			if line_count%2 == 0:
				l = l.split(',')
				lvl,flow_sp	,flow_q,res_sp,res_q = int(l[0]),float(l[4]),float(l[5]),float(l[6]),float(l[7])
				data[0,lvl].append((flow_q-flow_sp)/flow_q*100)
				data[1,lvl].append((res_q-res_sp)/res_q*100)
				# data[2,lvl].append(res_sp)
				# data[3,lvl].append(res_q)
			line_count += 1
	# Calculate the mean of each list, handling empty lists as zero
	average = np.array([np.mean(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)
	std_dev = np.array([np.std(lst) if len(lst) > 0 else 0 for lst in data.flatten()]).reshape(data.shape)

	labels = ['Flow','Residual']
	colors_tmp = ['#4169E1', '#228B22', '#DC143C', '#9932CC']
	band_colors = ['#ADD8E6', '#98FB98', '#F08080', '#E6E6FA']
	colors = ["royalblue", "forestgreen"]


	groupedbar(average.T,std_dev.T,f'Jitter Reduction (%)', 
		f'/home/bo/Dropbox/Research/NSDI24/images/qjitter_bar.eps',methods=labels,colors=colors,
		envs=[i for i in range(1,9)],ncol=1,sep=1,width=0.4,labelsize=24,lfsize=20,xlabel='Compression Level',legloc='best')

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
						psnr += ELF_adjust_PSNR[level]
						bpp *= ELF_adjust_bpp[level]
						if iteration>=1:
							psnr += se_adjust_psnr[level]
						else:
							psnr += sp_adjust_psnr[level]
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
		'# of Epochs','BPP',lbsize=24,lfsize=16,linewidth=4,markersize=8,ncol=1,annot_bpp_per_video=True,
		linestyles=line_styles,xticks=range(0,35,5),yticks=[.01,.02],bbox_to_anchor=(.57,1.02))
	line_plot(epoch_data,psnr_data,labels,line_colors,
		'/home/bo/Dropbox/Research/NSDI24/images/psnr_evolution_by_video.eps',
		'# of Epochs','PSNR (dB)',lbsize=24,lfsize=16,linewidth=4,markersize=8,ncol=1,annot_psnr_per_video=True,
		linestyles=line_styles,xticks=range(0,35,5),yticks=[32,34,36,38],bbox_to_anchor=(.58,0.67))



def plot_RD_tradeoff(methods = ['x264-veryslow','x265-veryslow','ScaleSpaceFlow','SSF-Finetuned','SSF-Collaborated']):
	num_methods = len(methods)
	ncol = 1
	bbox_to_anchor = (.27,.53)

	PSNRs = [[28.15,30.44,32.62,34.47,35.91,36.97,37.78,38.39],
				[29.48,31.74,33.79,35.46,36.73,37.57,38.10,38.62],
				[27.3895,28.7402,30.2928,32.4380,34.0819,36.0767,37.3006,40.3110],
				[34.90,37.94,37.80],
				[39.31,39.78,39.54,41.08],]
	bpps = [[0.0467,0.0691,0.1012,0.1467,0.2157,0.3417,0.6052,1.1411],
				[0.0524,0.0767,0.1112,0.1608,0.2435,0.3968,0.6952,1.2901],
				[0.0377,0.0439,0.0710,0.1069,0.1628,0.2226,0.2820,0.4685],
				[0.0219,0.0240,0.0904],
				[0.0110,0.0112,0.0232,0.0364],]

	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]

	line_plot(bpps,PSNRs,methods,colors,
			f'/home/bo/Dropbox/Research/NSDI24MCVC/images/psnr{num_methods}.eps',
			'BPP','PSNR (dB)',lbsize=24,lfsize=18,linewidth=4,
			ncol=ncol,markersize=8,bbox_to_anchor=bbox_to_anchor,use_arrow=False)

	bpps = [[0.0467,0.0691,0.1012,0.1467,0.2157,0.3417,0.6052,1.1411],
				[0.0524,0.0767,0.1112,0.1608,0.2435,0.3968,0.6952,1.2901],
				[0.0377,0.0439,0.0710,0.1069,0.1628,0.2226,0.2820,0.4685],
				[],
				[],]
	SSIMs = [[50.845,53.00,54.78,56.08,56.97,57.59,58.03,58.36],
			[51.79,53.81,55.47,56.68,57.49,57.98,58.17,58.40],
			[50.93,52.49,54.59,57.06,59.01,61.25,62.72,68.84],
			[],
			[]]

	line_plot(bpps,SSIMs,methods,colors,
			f'/home/bo/Dropbox/Research/NSDI24MCVC/images/ssim{num_methods}.eps',
			'BPP','SSIM (dB)',lbsize=24,lfsize=18,linewidth=4,
			ncol=ncol,markersize=8,)

def plot_QoE_cdf_breakdown(methods = ['Vesper','ELFVC','SSF','x264f','x264m','x264s','x265f','x265m','x265s'],
							folder = 'data'):
	# 16543747 bps=15.8Mbps
	# 4074145 mbps=3.9Mbps
	# colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
	selected_rows = [0] + [2+i for i in range(8)]
	num_methods = len(methods)
	if len(methods) == 3:
		selected_rows = [0,1,2]
	metric_list = []
	hw = 3090
	for k,metric in enumerate(['QoE','quality','rebuffer']):
		if k>0 and num_methods==3:break
		names = ['QoE','Quality','Rebuffer Rate']
		meanQoE_all = [];stdQoE_all = []
		matrices = []
		for trace in range(2):
			datafile = f'/home/bo/Dropbox/Research/NSDI24/{folder}/{metric}_{trace}_{hw}_1000.data'
			with open(datafile,'r') as f:
				line = f.readlines()[0]
			matrices += [np.array(eval(line))[selected_rows]]
		QoE_max = max(matrices[0].max(),matrices[1].max())
		QoE_min = min(matrices[0].min(),matrices[1].min())
		for trace in range(2):
			# QoE_min,QoE_max = QoE_matrix.min(),QoE_matrix.max()
			QoE_matrix = (matrices[trace] - QoE_min) / (QoE_max - QoE_min) 
			if k == 0 and num_methods!=3:
				bbox_to_anchor = (.8,.9) if trace == 0 else (0.17,1)
				arrow_coord = (0.1,0.5) if trace == 0 else (.8,.1)

				measurements_to_cdf(QoE_matrix,f'/home/bo/Dropbox/Research/NSDI24/images/{metric}cdf_{trace}_{hw}_{num_methods}.eps',methods,linestyles=linestyles,
					colors=colors,bbox_to_anchor=bbox_to_anchor,lfsize=17,ncol=1,lbsize=24,xlabel=f'Normalized {names[k]}',use_arrow=True,arrow_coord=arrow_coord,arrow_rotation=180)
			meanQoE = QoE_matrix.mean(axis=1)
			stdQoE = QoE_matrix.std(axis=1)
			meanQoE_all += [meanQoE]
			stdQoE_all += [stdQoE]
		meanQoE_all = np.stack(meanQoE_all).reshape(2,-1)
		# print(meanQoE_all.tolist())
		stdQoE_all = np.stack(stdQoE_all).reshape(2,-1)
		ncol = 1
		bbox_to_anchor = (1.22,1.05) if num_methods == 9 else (0.3,1)
		labelsize = 24
		lfsize = 16 if len(methods)==9 else 24
		width = 0.1 if len(methods)==9 else 0.2
		use_arrow=False;arrow_coord=(0,0);arrow_rotation=0
		if k<=1:
			use_arrow=True; arrow_coord = (1,0.65) if num_methods == 9 else (1.5,.25); arrow_rotation=-90
		elif k==2:
			use_arrow=True; arrow_coord = (1.25,0.33); arrow_rotation=90
		groupedbar(meanQoE_all,stdQoE_all,f'Normalized {names[k]}', 
			f'/home/bo/Dropbox/Research/NSDI24/images/{metric}mean_{num_methods}.eps',methods=methods,colors=colors,use_arrow=use_arrow,arrow_coord=arrow_coord,arrow_rotation=arrow_rotation,
			envs=['Limited BW','Adequate BW'],ncol=ncol,sep=1,width=width,labelsize=labelsize,lfsize=lfsize,bbox_to_anchor=bbox_to_anchor,xlabel='',ratio=.7)
	# 	if True:
	# 		for line in meanQoE_all.tolist():
	# 			ours,t_top1,l_top1 = line[0],max(line[1:3]),max(line[3:])
	# 			m1,m2=(ours - t_top1)/t_top1,(ours - l_top1)/l_top1
	# 			metric_list += [[m1,m2]]
	# metric_list = np.array(metric_list)
	# print(metric_list)
	# print(metric_list.mean(axis=0))
	# print(metric_list[:3].mean(axis=0))
	# print(metric_list[3:].mean(axis=0))

def plot_encoding_speed():
	labels = ['Vesper','ELFVC','SSF','x264f','x264m','x264s','x265f','x265m','x265s']
	y = [0.013, 0.0069, 0.0058,0.004889435564435564, 0.005005499857285572, 0.005083814399885828, 0.0054054160125588695, 0.0058336038961038965, 0.006322325888397318] 
	y = 1/np.array(y).reshape(-1,1)
	# color = '#808080'
	groupedbar(y,None,'Frame Rate (fps)', 
		'/home/bo/Dropbox/Research/NSDI24/images/encoding_speed.eps',methods=['QoE'],colors=['#4f646f'],labelsize=24,ylim=(0,230),
		envs=labels,ncol=0,rotation=45,use_realtime_line=True,bar_label_dxdy=(-0.4,5),yticks=range(0,250,30))


def plot_quantization_impact():
	colors = ['#1f77b4', '#2ca02c']
	labels = ['Train','Test']

	# Create some sample data
	epoch_data = [[],[]]; psnr_data = [[],[]]

	with open(f'../NSDI_logs/ELFVC.quantization.log','r') as f:
		epoch=1
		for l in f.readlines():
			l = l.split(',')
			train_loss,val_loss,train_bpp,train_psnr,val_bpp,val_psnr = [float(x) for x in l]
			epoch_data[0] += [epoch*100]; epoch_data[1] += [epoch*100]
			psnr_data[0] += [train_loss]; psnr_data[1] += [val_loss]
			epoch += 1

	path = '/home/bo/Dropbox/Research/NSDI24/images/quantization_impact.eps'
	line_plot(epoch_data,psnr_data,labels,colors,path,
		'# of Epochs','Loss',lbsize=32,lfsize=28,linewidth=4,markersize=8,linestyles=linestyles,annot_loss=True,bbox_to_anchor=(0.2,.5),ncol=1)

def plot_content_impact():
	labels = ['BPP','PSNR']
	colors = ['#1f77b4', '#2ca02c']
	linewidth = 4
	markersize = 8
	lfsize = 28
	labelsize = 32

	y1 = []; y2 = []
	x = []
	with open(f'../NSDI_logs/ELFVC.content.log','r') as f:
		epoch=0
		for l in f.readlines():
			l = l.split(',')
			bpp,psnr = float(l[1]),float(l[2])
			if epoch%2==0:
				psnr += ELF_adjust_PSNR[0]
				bpp *= ELF_adjust_bpp[0]
				y1 += [bpp]; y2 += [psnr]
				x += [epoch//2+1]
			epoch += 1

	# Create a figure and axis objects
	fig, ax1 = plt.subplots()
	plt.grid(True, which='both', axis='both', linestyle='--')

	# Plot data on the first y-axis
	ax1.plot(x, y1, color = colors[0], marker = markers[0], 
					linestyle = linestyles[0], 
					label = labels[0], 
					linewidth=linewidth, markersize=markersize)
	ax1.set_xlabel('# of Epochs',fontsize=labelsize)
	ax1.set_ylabel('Bits Per Pixel', color = colors[0],fontsize=labelsize)
	ax1.tick_params('y', color = colors[0],labelsize=labelsize)
	ax1.legend(loc='center right',fontsize = lfsize,bbox_to_anchor=(1,0.4))
	reduction = int(np.round((-y1[-1] + y1[0])/y1[0]*100))
	ax1.text(x[-1]-10, y1[-1]+0.001, f"\u2193{reduction}%", ha="center", va="center", size=labelsize, color=colors[0])

	# Create a second y-axis sharing the same x-axis
	ax2 = ax1.twinx()

	# Plot data on the second y-axis
	ax2.plot(x, y2, color = colors[1], marker = markers[1], 
					linestyle = linestyles[1], 
					label = labels[1], 
					linewidth=linewidth, markersize=markersize)
	ax2.set_ylabel('PSNR (dB)', color = colors[1],fontsize=labelsize)
	ax2.tick_params('y', color = colors[1],labelsize=labelsize)
	ax2.legend(loc='center right',fontsize = lfsize,bbox_to_anchor=(1,0.6))
	inc = y2[-1] - y2[0]
	sign = '+' if inc>0 else ''
	ax2.text(x[-1]-10, y2[-1]-1, f"{sign}{inc:.1f} dB", ha="center", va="center", size=labelsize, color=colors[1])

	path = '/home/bo/Dropbox/Research/NSDI24/images/content_impact.eps'
	fig.savefig(path,bbox_inches='tight')



plot_RD_tradeoff()