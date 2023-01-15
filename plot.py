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
				xticks=None,yticks=None,ncol=None, yerr=None,markers=markers,
				use_arrow=False,arrow_coord=(0.4,30),ratio=None,bbox_to_anchor=(1.1,1.2),use_doublearrow=False,linestyles=None):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			if linestyles is not None:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					linestyle = linestyles[i], 
					label = label[i], 
					linewidth=2, markersize=8)
			else:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], 
					linewidth=2, markersize=8)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linewidth=2, markersize=8)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lfsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
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

def bar_plot(avg,std,label,path,color,ylabel,yticks=None):
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
		plt.yticks( yticks,fontsize=labelsize )
	xleft, xright = ax.get_xlim()
	ybottom, ytop = ax.get_ylim()
	ratio = 0.3
	ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
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

def plot_clustered_stacked(dfall, filename, labels=None, xlabel='', ylabel='',**kwargs):
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
                rect.set_hatch(hatches[i]) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_xlabel(xlabel, size=18)
    axe.set_ylabel(ylabel, size=18)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=hatches[i]))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[.01, 0.78], fontsize=18)
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[.01, 0.48], fontsize=18) 
    axe.add_artist(l1)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    return axe
y = [[0.00922982069896534, 0.006790553347673267, 0.006487410833748678, 0.00638540335057769, 0.006372610218822956, 0.006348891967597106, 0.00633210981364495, 0.006320042988227215, 0.006351746211294085, 0.006317875129170716, 0.00633586478123272, 0.006272579183375152, 0.006262545791776994, 0.006247331099751006, 0.008030653400346636, 0.0067625498624693135, 0.007275365640902344, 0.007230396132864473, 0.007173292162999706, 0.007691734950058163, 0.007068861156724216, 0.007030596181919629, 0.006992190413217506, 0.007175724966994797, 0.006933906632009894, 0.006917426577106548, 0.007457253626220067, 0.006851207168074324, 0.007256177755409916, 0.006814253919680293, 0.007077082338923168, 0.006519024362205528],
[0.00548606279771775, 0.003431607753736898, 0.003367362101562321, 0.0032660109223797916, 0.0032579949405044314, 0.0032615564162066825, 0.003248569585515985, 0.003252119125681929, 0.0032554184992073312, 0.0032561225502286107, 0.003254697672938081, 0.0032521237172962476, 0.003259076938355485, 0.003262035650134619, 0.003373917040104667, 0.0033532534245750865, 0.003665525564814315, 0.003629065088332734, 0.0036002829211371897, 0.003569347424781881, 0.0035456652337286084, 0.003520996890835125, 0.003503622317119785, 0.003489803049887996, 0.003464867772068828, 0.0034628936962690206, 0.003461235892717485, 0.0034616328252013774, 0.003463102420310265, 0.0034601145163954544, 0.0034547646582547215, 0.003458931896966533],
[0.003717885306105018, 0.00378559454693459, 0.0029451658677620194, 0.002287238574353978, 0.0021216597803868356, 0.0018778003499998399, 0.001805677085316607, 0.0016600270493654535, 0.0016485498442004125, 0.0018172579305246472, 0.001782713045196777, 0.001552196857907499, 0.0015505399769888475, 0.0014956279208750596, 0.0017010874068364502, 0.0016520481251063757, 0.0016524716533775276, 0.0016496013558935374, 0.0016156969739026144, 0.0015830932295648379, 0.0015543500000300507, 0.0015240510454697703, 0.0015069524610779531, 0.0014778931998686556, 0.0014601151081733404, 0.001452930057600427, 0.0014340499480668869, 0.0014463814607422268, 0.0014186677828045753, 0.0014044352100851636, 0.0014014062419113133, 0.0013782492813334101],
[0.001617308601271361, 0.0008120687038172036, 0.0005483420643334588, 0.0005079873488284647, 0.0005018384009599686, 0.0004983724657601367, 0.0008471226569132081, 0.0009564759631757625, 0.0008940812213242882, 0.0009114686306566, 0.000936339036773213, 0.001327210333935606, 0.0012629704773784258, 0.0012071548503757056, 0.002202924427110702, 0.0021127071246155537, 0.0023262343410512102, 0.0022568482058381457, 0.002193114152300711, 0.002125761549687013, 0.0020859022663595773, 0.002040736749768257, 0.002004164347991995, 0.001952377845494387, 0.002807462880387902, 0.002766273457718154, 0.0027156622666451666, 0.002700615332079386, 0.0026588536758810795, 0.0026189185166731474, 0.002587619999934348, 0.002559150093657081],
[0.04257806269999946, 0.010202710700002626, 0.009899263466665312, 0.009842876325001271, 0.009643430019998504, 0.009574637699999281, 0.012225854628570688, 0.010813954025000783, 0.012988881333333464, 0.010087133600000017, 0.009947951745454537, 0.007765789866667205, 0.007750876999999981, 0.007672330157142986, 0.009972712186666588, 0.008837017281250326, 0.009255418358823, 0.009539506855555322, 0.009153754742105415, 0.009696313640000084, 0.009048943466667056, 0.009793481418181604, 0.008984935139130318, 0.008720860791666495, 0.009505605196000032, 0.009289199446153763, 0.009261210725925914, 0.009624589075000195, 0.009802052586207052, 0.01022242487333339, 0.010403062712902972, 0.010129123324999868],
[0.006785295899999255, 0.006383393999999498, 0.006349065266670096, 0.006348709024999266, 0.0063338079400000425, 0.006364333816666582, 0.006357241442857944, 0.006405456499999218, 0.0063724898999996385, 0.006249795739998945, 0.006270770699999998, 0.006256334116667025, 0.0063906088076928075, 0.0062817133142849405, 0.006283685240000523, 0.006276489218750214, 0.00630739382352939, 0.006295962316666722, 0.006311582421053094, 0.006286009589999822, 0.006306686014285593, 0.0063330741636362245, 0.00644377026521753, 0.006311391533333695, 0.0063195547679997625, 0.0063356534692310075, 0.006335374966666525, 0.006368552392856941, 0.006537267789655022, 0.0065546236866669, 0.0065885786193546425, 0.006624035659374883],
[0.03153221699999449, 0.005321238850001464, 0.004258850266664164, 0.003792742400000293, 0.003498925119999967, 0.0031887290000004974, 0.0030446434714284415, 0.002970204599999704, 0.002893189377778072, 0.0027502299599996153, 0.002813017736363318, 0.002762990908333525, 0.002750083846153582, 0.0027200425428572676, 0.002535795380000157, 0.0024998971125000935, 0.0024728609529407713, 0.0024502755111111838, 0.002399094705263475, 0.002370758229999979, 0.0024631274904761214, 0.002425416354545653, 0.002407447430434588, 0.002372477824999919, 0.0023384842360001127, 0.0023482348923078706, 0.002328709574074192, 0.0024093287392860574, 0.0023540934206897845, 0.00235820824999981, 0.0024161395193547626, 0.0024032242218751778],
[0.0015194659000030697, 0.0011410751500022798, 0.0011056342000036541, 0.0010875189500012539, 0.0010679148799999894, 0.0010715819666662204, 0.0010803763142868254, 0.0011003961875005075, 0.0010780082222216455, 0.0011183491999997841, 0.0010848615090904704, 0.0010883464833331875, 0.0010849595076916347, 0.0010823724071428841, 0.001102286493333698, 0.0010992839875001437, 0.0010898636941172246, 0.0011219940388893215, 0.0010983747631580867, 0.0011023434399999132, 0.0011143495333336823, 0.0011060634454545173, 0.0011192045260866943, 0.0011179709208334998, 0.0012516376720000152, 0.001241478696153728, 0.0012600432333333292, 0.001331905239285496, 0.0013047078827584896, 0.0013295708433334614, 0.0013328597677422472, 0.0013806849781250463]]
y = np.array(y)*1000
# y[2] += y[3]
# y[6] += y[7]
y = y[[5,6,7,1,2,3,],:16]

x = [range(1,y.shape[1]+1) for _ in range(y.shape[0])]

colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
line_plot(x,y,['Frame Prediction','Motion Codec','Residual Codec',2,3,4],colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/mot_parallel.eps',
		'Batch Size','Avg. Processing Time (ms)',ncol=1,legloc='best',lbsize=24,lfsize=18,bbox_to_anchor=None,xticks=range(0,17,2),
		use_doublearrow=False)
exit(0)

labels_tmp = ['DVC','RLVC','x264f','x264m','x264s','x265f','x265m','x265s']
y = [9.40291382e-01, 2.99550380e+00, 2.88042785e-03, 4.20175081e-03,3.68101004e-03, 3.08573793e-03, 2.47046928e-03, 2.46554348e-03] 
yerr = [0.07193951, 0.09499966, 0.00655925, 0.01298857, 0.01017801, 0.00530592,0.00158061, 0.00118021]
y = np.array(y).reshape(-1,1);yerr = np.array(yerr).reshape(-1,1)
groupedbar(y,yerr,'Rebuffer Rate', 
	'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/mot_rebuffer.eps',methods=['QoE'],colors=['#4f646f'],
	envs=labels_tmp,ncol=0,labelsize=24,rotation=45)


# RD tradeoff all
x640960_PSNR = [[33.916125480707116, 35.48593550438171, 37.016464507544086, 38.199682581079344, 39.118539617969084, 39.627869858965646, 40.20787336157038], [32.543478441762396, 34.573014674724995, 36.19663649696213, 37.28057779941882, 38.58632463985866, 39.22430920219802, 39.87904687051649], [32.65380431245734, 34.55438403483038, 36.062224239974356, 37.17589424849748, 37.86222585312255, 38.50796908617734, 39.215505181492624, 40.038787832031474], [31.74693045439896, 33.41208141047756, 34.92100008336695, 36.22079029021326, 37.261527250339455, 38.0560033669124, 38.6483168490045], [33.07011698223614, 34.53811576959493, 35.80980415825363, 36.87108862935961, 37.72481086704281, 38.402927373196334, 38.884062338780446], [32.843874187021704, 34.38099358012745, 35.724521354719116, 36.837054323364086, 37.71883388928005, 38.40001107048202, 38.890123292520926], [32.79842881913428, 34.26208116958191, 35.552305886557285, 36.645252529557766, 37.536240949259174, 38.228766637367684, 38.73681622856743], [32.75936338379904, 34.18578829179396, 35.46085622951343, 36.518808956746454, 37.36077980942779, 38.02671724671012, 38.53779179423482], [32.8636163464793, 34.33908462834049, 35.645658467080324, 36.718912166315356, 37.55216830355542, 38.19651269031453, 38.677899544531996]]
x640960_bpp = [[0.09785748001998001, 0.1369725024975025, 0.19493502747252747, 0.28089705294705297, 0.3879853021978022, 0.5577062437562439, 0.7615622627372628], [0.0635721028971029, 0.09370483266733268, 0.13689532967032966, 0.19788029470529467, 0.5477026098901099, 0.7162948801198801, 0.9466557942057943], [0.05500911588411588, 0.08862457542457543, 0.13366452297702297, 0.19711898101898104, 0.272768469030969, 0.39096426073926077, 0.5724937562437562, 0.8423756368631369], [0.051151173826173825, 0.08479610389610388, 0.14546853146853148, 0.26021663336663337, 0.47320390859140854, 0.8407709665334665, 1.4137036463536463], [0.06028671328671329, 0.09762492507492508, 0.15979454295704296, 0.26925576923076927, 0.46507764735264734, 0.8332161088911089, 1.4514770479520478], [0.05758936063936063, 0.09309536713286713, 0.15230112387612385, 0.25873106893106895, 0.44751312437562435, 0.810325024975025, 1.4410407717282716], [0.052209752747252744, 0.08867724775224775, 0.15350600649350649, 0.27385037462537465, 0.5003003621378621, 0.9180783216783216, 1.5802436813186813], [0.042274475524475524, 0.06967703546453546, 0.11463427822177823, 0.19173451548451548, 0.32591756993006993, 0.5685101148851149, 0.989252072927073], [0.04214488011988012, 0.07000197302697303, 0.11528716283716285, 0.19448647602397603, 0.3312023601398601, 0.5897757242757242, 1.0536495129870131]]
labels_tmp = ['Ours','DVC','RLVC','x264f','x264m','x264s','x265f','x265m','x265s']
colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
line_plot(x640960_bpp,x640960_PSNR,labels_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/RD_tradeoff.eps',
		'Bit Per Pixel','PSNR (dB)',markers=markers,ncol=2,ratio=None,bbox_to_anchor=(1,.1),legloc='lower right',lbsize=18,lfsize=16)


# motivation
UPSNRs = [[32.543478441762396, 34.573014674724995, 36.19663649696213, 37.28057779941882, 38.72632463985866, 39.22430920219802, 39.87904687051649], 
[32.65380431245734, 34.55438403483038, 36.062224239974356, 37.17589424849748, 37.86222585312255, 38.50796908617734, 39.215505181492624, 40.038787832031474], 
[31.74693045439896, 33.41208141047756, 34.92100008336695, 36.22079029021326, 37.261527250339455, 38.0560033669124, 38.6483168490045], 
[33.07011698223614, 34.53811576959493, 35.80980415825363, 36.87108862935961, 37.72481086704281, 38.402927373196334, 38.884062338780446], 
[32.843874187021704, 34.38099358012745, 35.724521354719116, 36.837054323364086, 37.71883388928005, 38.40001107048202, 38.890123292520926], 
[32.79842881913428, 34.26208116958191, 35.552305886557285, 36.645252529557766, 37.536240949259174, 38.228766637367684, 38.73681622856743], 
[32.75936338379904, 34.18578829179396, 35.46085622951343, 36.518808956746454, 37.36077980942779, 38.02671724671012, 38.53779179423482], 
[32.8636163464793, 34.33908462834049, 35.645658467080324, 36.718912166315356, 37.55216830355542, 38.19651269031453, 38.677899544531996]]
Ubpps = [[0.0635721028971029, 0.09370483266733268, 0.13689532967032966, 0.19788029470529467, 0.5477026098901099, 0.7162948801198801, 0.9466557942057943], 
[0.05500911588411588, 0.08862457542457543, 0.13366452297702297, 0.19711898101898104, 0.272768469030969, 0.39096426073926077, 0.5724937562437562, 0.8423756368631369], 
[0.051151173826173825, 0.08479610389610388, 0.14546853146853148, 0.26021663336663337, 0.47320390859140854, 0.8407709665334665, 1.4137036463536463], 
[0.06028671328671329, 0.09762492507492508, 0.15979454295704296, 0.26925576923076927, 0.46507764735264734, 0.8332161088911089, 1.4514770479520478], 
[0.05758936063936063, 0.09309536713286713, 0.15230112387612385, 0.25873106893106895, 0.44751312437562435, 0.810325024975025, 1.4410407717282716], 
[0.052209752747252744, 0.08867724775224775, 0.15350600649350649, 0.27385037462537465, 0.5003003621378621, 0.9180783216783216, 1.5802436813186813], 
[0.042274475524475524, 0.06967703546453546, 0.11463427822177823, 0.19173451548451548, 0.32591756993006993, 0.5685101148851149, 0.989252072927073], 
[0.04214488011988012, 0.07000197302697303, 0.11528716283716285, 0.19448647602397603, 0.3312023601398601, 0.5897757242757242, 1.0536495129870131]]

# for i in range(2):
# 	for j in range(2,8):
# 		bdrate = BD_RATE(np.array(Ubpps[i]),UPSNRs[i],np.array(Ubpps[j]),UPSNRs[j])
# 		print(bdrate)


colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
labels_tmp = ['DVC','RLVC','x264f','x264m','x264s','x265f','x265m','x265s']
line_plot(Ubpps,UPSNRs,labels_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/mot_RDtradeoff.eps',
		'Bit Per Pixel','PSNR (dB)',use_arrow=True,arrow_coord=(0.15,39.2),lbsize=24,lfsize=20,ncol=2,bbox_to_anchor=None)

datafile = f'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/QoE_0_1080_999.data'
with open(datafile,'r') as f:
	line = f.readlines()[0]
QoE_matrix = eval(line)
QoE_matrix = np.array(QoE_matrix)
QoE_min,QoE_max = QoE_matrix.min(),QoE_matrix.max()

y = [34.981735761479754, 33.06845540139495, 34.70190373585274, 35.411025852453434, 35.41881587778906, 35.207934780186314, 35.61876831087406, 35.76696244352575] 
yerr = [0.6939123489018372, 0.6456513214873756, 0.6517857008913301, 0.5864919879466348, 0.6076016211454529, 0.5619422880146273, 0.5154889680568497, 0.5192746802378674] 
y = np.array(y).reshape(-1,1);yerr = np.array(yerr).reshape(-1,1)
y = (y - QoE_min) / (QoE_max - QoE_min)
yerr /= (QoE_max - QoE_min)
groupedbar(y,yerr,'Normalized QoE', 
	'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/mot_QoEmean.eps',methods=['QoE'],colors=['#4f646f'],
	envs=labels_tmp,ncol=0,labelsize=24,ylim=(0,1),rotation=45)

y = [0.0382, 0.05810000000000001, 0.004889435564435564, 0.005005499857285572, 0.005083814399885828, 0.0074054160125588695, 0.0058336038961038965, 0.006322325888397318] 
y = 1/np.array(y).reshape(-1,1)
# Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
groupedbar(y,None,'FPS', 
	'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/mot_FPS.eps',methods=['QoE'],colors=['#4f646f'],labelsize=24,ylim=(0,230),
	envs=labels_tmp,ncol=0,rotation=45,use_realtime_line=True,bar_label_dxdy=(-0.4,5),yticks=range(0,250,30))


########################ABLATION####################################
# UVG
ab_labels = ['Default','w/o TS','Linear','One-hop']
bpps = [[0.12,0.18,0.266,0.37],
		[0.12,0.20,0.30,0.41],
        [0.10,0.15,0.23,0.33],
		[0.11,0.17,0.27,0.41],
		]
PSNRs = [[30.63,32.17,33.52,34.39],
		[29.83,31.25,32.74,34.05],
        [29.33,31.15,32.76,33.74],
		[29.77,31.62,32.99,33.92],
		]
line_plot(bpps,PSNRs,ab_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/ablation_rdtradeoff.eps',
		'Bit Per Pixel','PSNR (dB)',use_arrow=True,arrow_coord=(0.15,33),lbsize=24,lfsize=24,
		xticks=[.1,.2,.3,.4],yticks=range(30,35))

ab_labels = ['Default','w/o TS','Linear','One-hop']
fps_avg_list = [32.21, 32.63, 16.17, 32.98]
bar_plot(fps_avg_list,None,ab_labels,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/ablation_speed.eps',
		'#4f646f','FPS',yticks=range(0,40,10))

##############################High Bandwidth#############################
# 16543747 bps=15.8Mbps
# 4074145 mbps=3.9Mbps
labels_tmp = ['Ours','DVC','RLVC','x264f','x264m','x264s','x265f','x265m','x265s']
colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
for trace in range(2):
	meanQoE_all = [];stdQoE_all = []
	for hw in [1080,2080,3090]:
		datafile = f'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/QoE_{trace}_{hw}_1000.data'
		with open(datafile,'r') as f:
			line = f.readlines()[0]
		QoE_matrix = eval(line)
		QoE_matrix = np.array(QoE_matrix)
		QoE_min,QoE_max = QoE_matrix.min(),QoE_matrix.max()
		QoE_matrix = (QoE_matrix - QoE_min) / (QoE_max - QoE_min) 
		measurements_to_cdf(QoE_matrix,f'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/QoEcdf_{trace}_{hw}.eps',labels_tmp,linestyles=linestyles,
			colors=colors_tmp,bbox_to_anchor=(.14,1.02),lfsize=16,ncol=1)
		meanQoE = QoE_matrix.mean(axis=1)
		stdQoE = QoE_matrix.std(axis=1)
		meanQoE_all += [meanQoE]
		stdQoE_all += [stdQoE]
	meanQoE_all = np.stack(meanQoE_all).reshape(3,9)
	stdQoE_all = np.stack(stdQoE_all).reshape(3,9)
	groupedbar(meanQoE_all,stdQoE_all,'Normalized QoE', 
		f'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/QoEmean_{trace}.eps',methods=labels_tmp,colors=colors_tmp,
		envs=['1080','2080','3090'],ncol=1,sep=1,width=0.1,labelsize=18,lfsize=16,bbox_to_anchor=(1.22,1.05),xlabel='Hardware',ratio=.7)


df1 = pd.DataFrame([[29.7,19.5],[46,32.4]],
                   index=["RTX 2080 Ti", "GTX 1080 Ti"],
                   columns=["Encoding", "Decoding"])
df2 = pd.DataFrame([[39,28],[0051.9,40.2]],
                   index=["RTX 2080 Ti", "GTX 1080 Ti"],
                   columns=["Encoding", "Decoding"])
df3 = pd.DataFrame([[64.4,52.6],[79.0,63.2]],
                   index=["RTX 2080 Ti", "GTX 1080 Ti"],
                   columns=["Encoding", "Decoding"])

plot_clustered_stacked([df1, df2, df3],'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/encdec.eps',labels=['Ours','DVC','RLVC'],xlabel='Hardware',ylabel='Millisecond')

bit_dist = [0.025, 0.078,
0.033, 0.106,
0.045, 0.150,
0.063, 0.217,
0.026, 0.023,
0.034, 0.040,
0.046, 0.063,
0.068, 0.100,
0.016, 0.019,
0.025, 0.031,
0.034, 0.050,
0.050, 0.081]
bit_dist = np.array(bit_dist).reshape(12,2)

df1 = pd.DataFrame(bit_dist[:4],
                   index=["256", "512",'1024','2048'],
                   columns=["Motion", "Residual"])
df2 = pd.DataFrame(bit_dist[4:8],
                   index=["256", "512",'1024','2048'],
                   columns=["Motion", "Residual"])
df3 = pd.DataFrame(bit_dist[8:],
                   index=["256", "512",'1024','2048'],
                   columns=["Motion", "Residual"])

plot_clustered_stacked([df1, df2, df3],'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/bits_dist.eps',labels=['ELVC','DVC','RLVC'],xlabel='$\lambda$',ylabel='Bit per pixel')


y = [[0.0310,0.0382,0.0581,],
[0.0195,0.028,0.0526,],
[0.010,0.01,0.012,],]

y = 1/np.array(y)
labels_tmp = ['Ours','DVC','RLVC','x264f','x264m','x264s','x265f','x265m','x265s']
additional_y = [0.004889435564435564, 0.005005499857285572, 0.005083814399885828, 
0.0074054160125588695, 0.0058336038961038965, 0.006322325888397318]
additional_y = 1/np.array(additional_y)
colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']

groupedbar(y,None,'Frame Rate (fps)', 
	'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/FPS_all.eps',methods=labels_tmp,colors=colors_tmp,
	envs=['1080','2080','3090'],ncol=1,sep=1,width=0.3,labelsize=18,lfsize=16,
	additional_y=additional_y,bbox_to_anchor=(.15,1.1),xlabel='Hardware')

#######################ERROR PROP########################
eplabels = ['Ours','DVC','RLVC'] # UVG,r=2048
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
ELVC_error = [
[30.324174880981445, 30.538787841796875, 30.67068862915039, 30.49388313293457, 30.715848922729492, 31.25832176208496, 30.150089263916016, 31.260454177856445, 30.7042293548584, 30.511245727539062, 30.683881759643555, 30.523727416992188, 30.34703254699707],
[31.795574188232422, 32.01017379760742, 32.13482666015625, 32.016048431396484, 32.25210189819336, 32.848426818847656, 32.09651565551758, 32.85036849975586, 32.24248123168945, 32.039222717285156, 32.14407730102539, 31.9993953704834, 31.81879997253418],
[33.16758728027344, 33.39623260498047, 33.38203430175781, 33.413753509521484, 33.68014144897461, 34.21955108642578, 33.17623519897461, 34.22688674926758, 33.68741989135742, 33.44963455200195, 33.401241302490234, 33.39918518066406, 33.20851516723633],
[34.112457275390625, 34.34897232055664, 34.174808502197266, 34.358848571777344, 34.64569854736328, 35.04032516479492, 33.5637092590332, 35.05060958862305, 34.66664505004883, 34.40544891357422, 34.20201110839844, 34.366241455078125, 34.17375183105469],
]
ytick_list = [range(29,32),range(31,34),range(32,35),range(33,36)]
for i in range(4):
    PSNRs = [ELVC_error[i],DVC_error[i],RLVC_error[i]]
    ylabel = 'PSNR (dB)' if i==0 else ''
    legloc = 'lower center' if i==0 else 'best'
    line_plot(frame_loc,PSNRs,eplabels,colors,
            f'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/error_prop_{i}.eps',
            'Frame Index',ylabel,xticks=range(1,14),yticks=ytick_list[i],lbsize=24,lfsize=18,
           	legloc=legloc)

######################SCALABILITY##########################
decompt =[ [0.02196,0.01354,0.0100,0.00785,0.00689],
[0.014,0.011,0.010,0.009,0.009],
[0.013,0.012,0.012,0.012,0.012],
[0.03416528925619835, 0.02003305785123967, 0.0195, 0.01668595041322314, 0.0161900826446281],
[0.03349720670391061, 0.025944134078212295, 0.028, 0.026748603351955308, 0.026837988826815644],
[0.057956135770234986, 0.060977545691906006, 0.0526, 0.053080678851174935, 0.05349268929503916],
[0.060496067755595885, 0.04007614467488227, 0.03473126682295737, 0.031039031582214632, 0.030088761847449977], 
[0.04193311667889715, 0.039980009995002494, 0.0388651379712398, 0.038204393505253106, 0.03818615751789976], 
[0.04637143519591931, 0.05238344683080147, 0.05676979846721544, 0.058114194391980234, 0.05871128724497285],]
decompt = 1/np.array(decompt)
# motivation show duration
scalability_labels = ['Ours (3090)','DVC (3090)','RLVC (3090)','Ours (2080)','DVC (2080)','RLVC (2080)','Ours (1080)','DVC (1080)','RLVC (1080)']
show_indices = [0,1,5,13,29] # 1,2,6,14,30
GOP_size = [[(i+1)*2+1 for i in show_indices] for _ in range(len(scalability_labels))]
colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
line_plot(GOP_size,decompt,scalability_labels,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/scalability_fps.eps',
		'GOP Size','FPS',ncol=3,legloc='upper center',bbox_to_anchor=(0.45,-.16),lbsize=18,lfsize=14,ratio=.6,
		xticks=range(0,61,10),yticks=range(0,151,30),linestyles=linestyles)

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
sc_labels = ['GOP=3','GOP=5','GOP=13','GOP=29','GOP=61']
line_plot(Sbpps,SPSNRs,sc_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23-VC/images/scalability_rdtradeoff.eps',
		'Bit Per Pixel','PSNR (dB)',use_arrow=True,arrow_coord=(0.15,34),lbsize=24,lfsize=24,
		xticks=[0.1,0.2,.3,.4,.5,.6,.7],yticks=range(30,35))

