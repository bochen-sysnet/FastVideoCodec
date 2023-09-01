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
				use_arrow=False,arrow_coord=(60,0.6),markersize=8,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_probarrow=False,
				rotation=None,use_resnet56_2arrow=False,use_resnet56_3arrow=False,use_resnet56_4arrow=False,use_resnet50arrow=False,use_re_label=False,
				use_throughput_annot=False,use_connarrow=False,lgsize=None,oval=False,scatter_soft_annot=False,markevery=1,annot_aw=None,
				fill_uu=False,failure_annot=False,failure_repl_annot=False,use_dcnbw_annot=False,latency_infl_annot=False,ur_annot=False):
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
		if oval:
			width = np.std(xx); height = np.std(yy)
			xi = np.mean(xx); yi = np.mean(yy)
			if width>0 and height>0:
				ellipse = Ellipse((xi, yi), width, height, edgecolor=None, facecolor=color[i],label = label[i], )
				ax.add_patch(ellipse)
				handles.append(ellipse)
				# plt.errorbar(xi, yi, yerr=height, color = color[i], label = label[i], linewidth=0, capsize=0, capthick=0)
			else:
				error_bar = plt.errorbar(xi, yi, yerr=height, color = color[i], label = label[i], linewidth=4, capsize=8, capthick=3)
				handles.append(error_bar)
		else:
			if yerr is None:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery)
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
	if fill_uu:
		ax.text(0.14, 85, "Futile\nReplica", ha="center", va="center", size=lgsize,fontweight='bold')
		plt.fill_between(XX[0], YY[0], YY[1], where=(YY[0] >= YY[1]), interpolate=True, color='grey', alpha=0.9)
	if use_connarrow:
		ax.annotate(text='', xy=(XX[0][5],YY[0][5]), xytext=(XX[0][5],0), arrowprops=dict(arrowstyle='|-|',lw=4))
		ax.text(XX[0][5]+7, YY[0][5]/2, "50% loss", ha="center", va="center", rotation='vertical', size=lbsize,fontweight='bold')
	if use_probarrow:
		ax.annotate(text='', xy=(XX[0][1],YY[0][1]), xytext=(XX[1][1],YY[1][1]), arrowprops=dict(arrowstyle='<->',lw=4))
		if YY[0][1]/YY[1][1]>10:
			ax.text(
			    XX[0][1]+0.1, (YY[0][1]+YY[1][1])/2, f"{YY[0][1]/YY[1][1]:.1f}"+r"$\times$", ha="center", va="center", rotation='vertical', size=44,fontweight='bold')
		else:
			ax.text(
			    XX[0][1]-0.07, (YY[0][1]+YY[1][1])/2, f"{YY[0][1]/YY[1][1]:.1f}"+r"$\times$", ha="center", va="center", rotation='vertical', size=44,fontweight='bold')
	if use_re_label:
		baselocs = [];parlocs = []
		for i in [1,2,3]:
			baselocs += [np.argmax(y[i]-y[0])]
			# parlocs += [np.argmax(y[i]-y[4])]
		for k,locs in enumerate([baselocs]):
			for i,loc in enumerate(locs):
				ind_color = '#4f646f'
				ax.annotate(text='', xy=(XX[0][loc],YY[0 if k==0 else 4,loc]), xytext=(XX[0][loc],YY[i+1,loc]), arrowprops=dict(arrowstyle='|-|',lw=5-k*2,color=ind_color))
				h = YY[k,loc]-5 if k==0 else YY[i+1,loc]+4
				w = XX[0][loc]-3 if k==0 else XX[0][loc]
				if k==0 and i==1:
					h-=1;w+=3
				if i==0:
					ax.text(w, h, '2nd', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
					ax.annotate(text='Consistency', xy=(45,67), xytext=(-5,65), size=22,fontweight='bold', arrowprops=dict(arrowstyle='->',lw=2,color='k'))
				elif i==1:
					ax.text(w, h-3, '3rd', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
				elif i==2:
					ax.text(w, h, '4th', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
	if scatter_soft_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			if i==0:
				ax.annotate('Most Reliable\nMost Computation', xy=(xx[i],yy[i]), xytext=(xx[i]-50,yy[i]+2),color = color[i], fontsize=lbsize-4,arrowprops=dict(arrowstyle='->',lw=2))
			elif i==1:
				ax.annotate('Least Computation\nMost Unreliable', xy=(xx[i],yy[i]), xytext=(xx[i]-5,yy[i]-4),color = color[i], fontsize=lbsize-4,arrowprops=dict(arrowstyle='->',lw=2))
	if use_throughput_annot:
		ax.annotate(text=f"$\u2191$"+'41%', xy=(XX[1][1],YY[1][1]), xytext=(0,0.8), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize+2,fontweight='bold')
	if ur_annot:
		ax.annotate(text=f"98.1% FR\nat 99% SR", xy=(99,98.1), xytext=((55,80)), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',color = color[0])
		
		ax.text(30,42, "More Success,\nmore futile\nreplicas", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.annotate(text="",xy=(50,30),xytext=(70,50), arrowprops=dict(arrowstyle='<-',lw=2),size=lgsize,fontweight='bold',)
		ax.text(70,5, "More replicas,\nhigher futile rate", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.annotate(text="",xy=(100,10),xytext=(100,35), arrowprops=dict(arrowstyle='<-',lw=2),size=lgsize,fontweight='bold',)
	if use_dcnbw_annot:
		ax.annotate(text="0.6 Gbps", xy=(XX[2][0], YY[2][0]), xytext=(XX[2][0]-30, YY[2][0]+0.01), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',)
		ax.annotate(text="0.1 Gbps", xy=(XX[2][4], YY[2][4]), xytext=(XX[2][4], YY[2][4]+0.01), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',)
	if latency_infl_annot:
		ax.text(76,9.6, "DCN latency\ndominates inflation", ha="center", va="center", size=lgsize,fontweight='bold',color=colors[1])
		ax.text(60,3.3, "Negligible DCN latency", ha="center", va="center", size=lgsize,fontweight='bold',color=colors[0])
	if failure_annot:
		ax.text(6, 74.5, "REACTIQ better", ha="center", va="center", size=lgsize,fontweight='bold')
		ax.text(0.0, 74.5, "Original\nbetter", ha="center", va="center", size=lgsize,fontweight='bold')
		plt.fill_between(XX[0], YY[0], YY[3], where=(YY[0] > YY[3]), interpolate=True, color="#307c9d", alpha=0.1)
		plt.fill_between(XX[0], YY[0], YY[3], where=(YY[0] <= YY[3]), interpolate=True, color="#f7546a", alpha=0.1)
		plt.fill_between(XX[2], YY[0], YY[2], where=(YY[0] <= YY[2]), interpolate=True, color="#ffffff", alpha=0.1)
		ax.annotate(text=f"4.1%", xy=(4.1,73.08), xytext=((6,70)), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',color = color[2])
		ax.annotate(text=f"1.2%", xy=(1.2,75.29), xytext=((1,73.5)), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',color = color[3])
	if failure_repl_annot:
		plt.fill_between(XX[0], YY[0], YY[1], where=(YY[0] <= YY[1]), interpolate=True, color="#ffab02", alpha=0.1)
		plt.fill_between(XX[0], YY[0], YY[2], where=(YY[0] <= YY[2]), interpolate=True, color= "#9dc51d", alpha=0.1)
		ax.annotate(text=f"11.8%", xy=(11.8,73.82), xytext=((11,75)), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',color = color[1])
		ax.annotate(text=f"15.1%", xy=(15.1,73.09), xytext=((15,74.5)), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold',color = color[2])
	if annot_aw is not None:
		if annot_aw == 0:
			ax.annotate(text='0.37% to max', xy=(0.4,97.28), xytext=(0.45,90), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		elif annot_aw == 1:
			ax.annotate(text='0.08% to max', xy=(0.4,75.21), xytext=(0.45,73.5), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		else:
			ax.annotate(text='0.7% inflation', xy=(0.4,0.7), xytext=(0.05,1.4), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
	
	if use_resnet50arrow:
		ax.annotate(text='', xy=(0,64), xytext=(40,64), arrowprops=dict(arrowstyle='<->',lw=2))
		ax.text(
		    20, 65, "Stage#0", ha="center", va="center", rotation='horizontal', size=lgsize,fontweight='bold')
		ax.annotate(text='', xy=(40,70), xytext=(160,70), arrowprops=dict(arrowstyle='<->',lw=2))
		ax.text(
		    100, 69, "Stage#1", ha="center", va="center", rotation='horizontal', size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.1', xy=(0,68), xytext=(-8,75), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.01', xy=(80,72.2), xytext=(40,74), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.001', xy=(120,75), xytext=(100,73), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		# for l,r,lr in [(0,40,0.1),(40,80,0.1),(80,120,0.01),(120,160,0.001)]:
		# 	ax.annotate(text='', xy=(l,64), xytext=(r,64), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		# 	ax.text(
		# 	    (l+r)/2, 64.5, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
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
				showaccbelow=False,showcompbelow=False,bw_annot=False,showrepaccbelow=False,breakdown_annot=False,frameon=True):
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


def plot_RD_tradeoff():
	ncol = 1
	bbox_to_anchor = (.27,.53)
	num_methods = 4
	bpps = [[[] for _ in range(num_methods)] for _ in range(5)]
	PSNRs = [[[] for _ in range(num_methods)] for _ in range(5)]
	SSIMs = [[[] for _ in range(num_methods)] for _ in range(5)]
	bw_saves = [[[] for _ in range(num_methods)] for _ in range(5)]
	filenames = ['MCVC-IA-OLFT.cat.log','MCVC-Original.avg.log','x264-veryslow.avg.log']
	pos_list = [0,2,3]
	for i,filename in enumerate(filenames):
		with open(filename, mode='r') as f:
			for l in f.readlines():
				l = l.split(',')
				if filename == 'MCVC-IA-OLFT.cat.log':
					cat,lvl,bpp,psnr0,ssim0,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6])
					r0,r_start,r_end = float(l[7]),float(l[8]),float(l[9])
					bpps[cat][1] += [bpp*1080*1920*views_of_category[cat]/1024/1024]
					PSNRs[cat][1] += [psnr0]
					SSIMs[cat][1] += [ssim0]
				else:
					cat,lvl,bpp,psnr,ssim = int(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
				if bpp > 0.2:
					continue
				bpp *= 1080*1920*views_of_category[cat]/1024/1024
				bpps[cat][pos_list[i]] += [bpp]
				PSNRs[cat][pos_list[i]] += [psnr]
				SSIMs[cat][pos_list[i]] += [ssim]
	methods = ['Ours','Ours (w/ OL)','SSF','x264']
	bd_all = [[] for _ in range(5)]
	for cat in range(5):
		for i in range(3):
			bd = BD_RATE(bpps[cat][i], PSNRs[cat][i], bpps[cat][-1], PSNRs[cat][-1])
			bd_all[cat] += [bd]
		line_plot(bpps[cat],PSNRs[cat],methods,colors,
				f'images/psnr_{cat}.pdf',
				'Bandwidth Usage (Mbps)','PSNR (dB)',lbsize=24,lgsize=20,linewidth=2,
				ncol=ncol,markersize=8,bbox_to_anchor=None,xticks=[0,0.5,1,1.5,2])
		line_plot(bpps[cat],SSIMs[cat],methods,colors,
				f'images/ssim_{cat}.pdf',
				'Bandwidth Usage (Mbps)','SSIM (dB)',lbsize=24,lgsize=20,linewidth=2,
				ncol=ncol,markersize=8,bbox_to_anchor=None,xticks=[0,0.5,1,1.5,2])
		
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
			ncol=1,markersize=8,bbox_to_anchor=(0.38,1.03),legloc=None,xlim=(0,.8))
	line_plot(bpps,si_list,methods,colors,
			f'images/si.pdf',
			'Bandwidth Usage (Mbps)','Sampling Interval',lbsize=24,lgsize=16,linewidth=2,
			ncol=1,markersize=8,bbox_to_anchor=None,legloc='best')
	envs = [0,1,2,3]
	groupedbar(np.array(bw_real_list).T,None,'Actual BW Impact (%)', 
		'images/bw_impact.pdf',methods=methods,labelsize=24,xlabel='Compression Levels',
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
	methods = ['GTX 1080', 'RTX 2080', 'RTX 3090']
	envs = np.array(bpp_list).mean(axis=1)
	envs = [f'{b:.2f}' for b in envs]
	groupedbar(np.array(psnr_list),None,'PSNR (dB)', 
		'images/psnr_vs_c2s.pdf',methods=methods,labelsize=24,xlabel='Bandwidth Usage (Mbps)',
		envs=envs,ncol=1,width=1./5,sep=1,legloc='best',lgsize=22,ylim=(30,34))
	
def plot_nv():
	psnr_list = [[] for _ in range(4)]
	nv_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.nv.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			nv,lvl,bpp,psnr = int(l[0]),int(l[1]),float(l[2]),float(l[3]),
			psnr_list[lvl] += [psnr]
			nv_list[lvl] += [nv]
	methods = ['Level=0', 1,2,3]
	line_plot(nv_list,psnr_list,methods,colors,
			f'images/psnr_vs_nv.pdf',
			'# of Views','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=8,bbox_to_anchor=None,legloc='best',)
	
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
	methods = ['Level=0', 1,2,3]
	line_plot(bw_list,psnr_list,methods,colors,
			f'images/psnr_vs_sr.pdf',
			'Bandwidth Usage (%)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=8,bbox_to_anchor=None,legloc='best',
			xticks=[-2,-1,0],
			xticklabel=[1,10,100])
	
def plot_si():
	psnr_list = [[] for _ in range(4)]
	bw_list = [[] for _ in range(4)]
	with open('MCVC-IA-OLFT.si.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			si,lvl,psnr = float(l[0]),int(l[1]),float(l[3])
			psnr_list[lvl] += [psnr]
			if si == 0:
				si = 10
			else:
				si = 1/si
			bw_list[lvl] += [np.log10(si)]
	methods = ['Level=0', 1,2,3]
	line_plot(bw_list,psnr_list,methods,colors,
			f'images/psnr_vs_si.pdf',
			'Bandwidth Usage (%)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=2,markersize=8,bbox_to_anchor=None,legloc='best',
			xticks = [-3,-2,-1,0,1],
			xticklabel= [0.1,1,10,100,'Base'])

	
def plot_mps():
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)] 
	with open('MCVC-IA-OLFT.mps.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			mps,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			x_list[lvl] += [mps*5]
			psnr_list[lvl] += [psnr]
	methods = ['Level=0', 1,2,3]
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_mps.pdf',
			'Relative Cache Size (%)','PSNR (dB)',lbsize=21,lgsize=18,linewidth=2,
			ncol=2,markersize=8,bbox_to_anchor=None,legloc='best',
			xticks=[5,25,50,75,100],
			xticklabel=['#GOP=1',25,50,75,100])

def plot_dr():
	psnr_list = [[] for _ in range(4)]
	x_list = [[] for _ in range(4)] 
	with open('MCVC-IA-OLFT.dr.log', mode='r') as f:
		for l in f.readlines():
			l = l.split(',')
			dr,lvl,bpp,psnr,ssim = float(l[0]),int(l[1]),float(l[2]),float(l[3]),float(l[4])
			x_list[lvl] += [dr*100]
			psnr_list[lvl] += [psnr]
	methods = ['Level=0', 1,2,3]
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_dr.pdf',
			'Streaming Duration (%)','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=2,markersize=8,bbox_to_anchor=(0.73,.73),legloc='best',)
	
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
	methods = ['Level=0', 1,2,3]
	line_plot(x_list,psnr_list,methods,colors,
			f'images/psnr_vs_sisr.pdf',
			'Spatial Sampling Ratio','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=8,bbox_to_anchor=None,legloc='best',
			xticks = [-4,-3,-2,-1,0],
			xticklabel= [0,0.001,0.01,0.1,1])
	
def plot_nv_vs_year():
	# dataset:
	# assembly 101, 8+4 views, 2022

	# codec
	pass

plot_dr()
plot_mps()
plot_si()
exit(0)
plot_sr()
plot_sisr()
plot_vary_compute()
plot_bw()
plot_RD_tradeoff()
plot_nv()