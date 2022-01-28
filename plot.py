#!/usr/bin/python

import numpy as np
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
labels = ['LSVC','H.264','H.265','DVC','RLVC']
markers = ['o','P','s','>','D','^']
linestyles = ['solid','dotted','dashed','dashdot', (0, (3, 5, 1, 5, 1, 5))]


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
				xticks=None,yticks=None,ncol=None, yerr=None,
				use_arrow=False,arrow_coord=(0.4,30)):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				linestyle = linestyles[i], label = label[i], 
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
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=lbsize-8,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	plt.tight_layout()
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lfsize)
		else:
			plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol)
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
	hbar = ax.bar(ind, avg, width, color=color, \
		yerr=std, error_kw=dict(lw=1, capsize=1, capthick=1))
	ax.set_ylabel(ylabel, fontsize = labelsize)
	ax.set_xticks(ind,fontsize=labelsize)
	ax.set_xticklabels(label, fontsize = labelsize)
	ax.bar_label(hbar, fmt='%.2f', fontsize = labelsize)
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

	d = .015 # how big to make the diagonal lines in axes coordinates
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

######################PIPELINE##########################
#LSVC
[30.324174880981445, 30.538787841796875, 30.67068862915039, 30.493885040283203, 30.715839385986328, 31.258317947387695, 30.150089263916016, 31.260454177856445, 30.704235076904297, 30.511245727539062, 30.683883666992188, 30.52373695373535, 30.347036361694336]
[28.298402786254883, 29.32883071899414, 26.488296508789062, 28.39558219909668, 29.504722595214844, 28.867931365966797, 0.0, 28.858898162841797, 29.479257583618164, 28.426942825317383, 26.54940414428711, 29.309539794921875, 28.29562759399414]
[28.656272888183594, 29.642196655273438, 26.948925018310547, 28.73621940612793, 29.792583465576172, 29.28494644165039, 0.0, 29.288658142089844, 29.771686553955078, 28.774675369262695, 27.026559829711914, 29.634706497192383, 28.650615692138672]

[31.79556655883789, 32.010169982910156, 32.13483428955078, 32.01605224609375, 32.252105712890625, 32.848426818847656, 32.09651565551758, 32.850372314453125, 32.242488861083984, 32.039241790771484, 32.144073486328125, 31.9993953704834, 31.818790435791016]
[29.2729434967041, 30.47451400756836, 27.35369110107422, 29.36699867248535, 30.667308807373047, 30.300966262817383, 0.0, 30.276790618896484, 30.64506721496582, 29.374788284301758, 27.421789169311523, 30.45831871032715, 29.23392677307129]
[29.642555236816406, 30.805307388305664, 27.43917465209961, 29.723651885986328, 30.977153778076172, 30.32033920288086, 0.0, 30.323108673095703, 30.949195861816406, 29.71859359741211, 27.507112503051758, 30.794078826904297, 29.59737777709961]

[33.16758728027344, 33.39623260498047, 33.38203430175781, 33.413753509521484, 33.68014144897461, 34.219547271728516, 33.17623519897461, 34.22687911987305, 33.68741989135742, 33.449623107910156, 33.401241302490234, 33.39919662475586, 33.208526611328125]
[29.97351837158203, 31.370927810668945, 27.767627716064453, 30.074676513671875, 31.587181091308594, 31.054536819458008, 0.0, 31.03588104248047, 31.57195472717285, 30.092966079711914, 27.84052848815918, 31.368003845214844, 29.947742462158203]
[30.407732009887695, 31.75274658203125, 27.98227882385254, 30.49777603149414, 31.962841033935547, 31.130823135375977, 0.0, 31.135499954223633, 31.942415237426758, 30.512845993041992, 28.072826385498047, 31.765607833862305, 30.374088287353516]

[34.112464904785156, 34.348960876464844, 34.17481231689453, 34.358848571777344, 34.64569854736328, 35.04032516479492, 33.5637092590332, 35.05060958862305, 34.66664123535156, 34.40544128417969, 34.2020149230957, 34.36622619628906, 34.17374801635742]
[30.361879348754883, 31.887596130371094, 27.905567169189453, 30.454771041870117, 32.1118049621582, 31.319427490234375, 0.0, 31.30449867248535, 32.09374237060547, 30.479469299316406, 27.99410057067871, 31.888900756835938, 30.34747314453125]
[30.823848724365234, 32.28938293457031, 27.904136657714844, 30.906814575195312, 32.51226043701172, 30.9323787689209, 0.0, 30.935863494873047, 32.491764068603516, 30.917221069335938, 27.996685028076172, 32.30657958984375, 30.794898986816406]

# DVC
[29.173748016357422, 29.27092170715332, 29.37064552307129, 29.497413635253906, 29.661972045898438, 29.852237701416016, 30.150089263916016, 29.852378845214844, 29.661563873291016, 29.503948211669922, 29.374170303344727, 29.260009765625, 29.161935806274414]
[28.481109619140625, 28.577266693115234, 28.69678497314453, 28.772428512573242, 28.930997848510742, 29.2314510345459, 0.0, 29.223670959472656, 28.931238174438477, 28.82183837890625, 28.701223373413086, 28.59286880493164, 28.49413299560547]
[28.35502052307129, 28.450502395629883, 28.55092430114746, 28.612621307373047, 28.753910064697266, 28.990413665771484, 0.0, 28.98839569091797, 28.741500854492188, 28.65834617614746, 28.556934356689453, 28.46303367614746, 28.369247436523438]

[30.878501892089844, 30.991844177246094, 31.11358070373535, 31.2701416015625, 31.462112426757812, 31.68387794494629, 32.09651565551758, 31.678407669067383, 31.453899383544922, 31.26241111755371, 31.110883712768555, 30.966976165771484, 30.851383209228516]
[29.87420654296875, 29.990354537963867, 30.114185333251953, 30.209226608276367, 30.38852882385254, 30.745929718017578, 0.0, 30.724746704101562, 30.356836318969727, 30.25389862060547, 30.1245059967041, 29.99236297607422, 29.882112503051758]
[29.70517921447754, 29.816722869873047, 29.92607879638672, 29.994768142700195, 30.155187606811523, 30.442304611206055, 0.0, 30.4287109375, 30.122648239135742, 30.037315368652344, 29.9339656829834, 29.82136344909668, 29.711952209472656]

[32.132957458496094, 32.243408203125, 32.357173919677734, 32.50341796875, 32.68071746826172, 32.86759948730469, 33.17623519897461, 32.866546630859375, 32.6748161315918, 32.504634857177734, 32.3598747253418, 32.22612762451172, 32.110939025878906]
[30.806640625, 30.92320442199707, 31.03727912902832, 31.1124324798584, 31.257946014404297, 31.568897247314453, 0.0, 31.560964584350586, 31.228649139404297, 31.157615661621094, 31.036495208740234, 30.932708740234375, 30.81328773498535]
[30.63261604309082, 30.742660522460938, 30.837276458740234, 30.896705627441406, 31.00812530517578, 31.22210121154785, 0.0, 31.225732803344727, 30.971023559570312, 30.93768310546875, 30.838455200195312, 30.753334045410156, 30.642309188842773]

[32.98991775512695, 33.087459564208984, 33.17992401123047, 33.29349899291992, 33.42362594604492, 33.54410171508789, 33.5637092590332, 33.54086685180664, 33.418758392333984, 33.29288101196289, 33.18034362792969, 33.075042724609375, 32.984214782714844]
[31.387508392333984, 31.473073959350586, 31.574783325195312, 31.612934112548828, 31.725500106811523, 31.88429832458496, 0.0, 31.87765121459961, 31.704010009765625, 31.667377471923828, 31.58749771118164, 31.495058059692383, 31.41139793395996]
[31.19398307800293, 31.28240966796875, 31.35838508605957, 31.370620727539062, 31.45598602294922, 31.511009216308594, 0.0, 31.501174926757812, 31.42658805847168, 31.41568946838379, 31.364208221435547, 31.294601440429688, 31.21670150756836]

#RLVC
[28.934640884399414, 29.0654354095459, 29.188518524169922, 29.363454818725586, 29.605087280273438, 30.001672744750977, 30.150089263916016, 29.993770599365234, 29.60120391845703, 29.366622924804688, 29.19466209411621, 29.043128967285156, 28.91464614868164]
[28.252681732177734, 28.382244110107422, 28.518884658813477, 28.6552791595459, 28.917280197143555, 29.244461059570312, 0.0, 29.230913162231445, 28.90703010559082, 28.684768676757812, 28.525779724121094, 28.385425567626953, 28.257638931274414]
[28.231082916259766, 28.35358428955078, 28.465999603271484, 28.584169387817383, 28.787567138671875, 28.802413940429688, 0.0, 28.758514404296875, 28.77138900756836, 28.609603881835938, 28.474363327026367, 28.343585968017578, 28.236831665039062]

[30.781314849853516, 30.918954849243164, 31.047901153564453, 31.23015785217285, 31.494022369384766, 31.950214385986328, 32.09651565551758, 31.95067596435547, 31.49615478515625, 31.236324310302734, 31.0521240234375, 30.89522933959961, 30.75379753112793]
[29.768478393554688, 29.898950576782227, 30.04826545715332, 30.1702938079834, 30.465646743774414, 30.885608673095703, 0.0, 30.889318466186523, 30.46480369567871, 30.215803146362305, 30.044313430786133, 29.90804672241211, 29.778724670410156]
[29.678958892822266, 29.805009841918945, 29.914960861206055, 30.010356903076172, 30.22585678100586, 30.278287887573242, 0.0, 30.2508487701416, 30.222610473632812, 30.053380966186523, 29.915616989135742, 29.801546096801758, 29.694578170776367]

[32.05010986328125, 32.20680236816406, 32.35564041137695, 32.55833435058594, 32.83184051513672, 33.269229888916016, 33.17623519897461, 33.277217864990234, 32.842647552490234, 32.57870864868164, 32.374053955078125, 32.1959342956543, 32.042667388916016]
[30.808195114135742, 30.94692611694336, 31.095979690551758, 31.21013641357422, 31.491060256958008, 31.887453079223633, 0.0, 31.89170265197754, 31.485570907592773, 31.271921157836914, 31.107885360717773, 30.96190643310547, 30.832374572753906]
[30.683185577392578, 30.819730758666992, 30.93010902404785, 31.01856803894043, 31.226238250732422, 31.12099838256836, 0.0, 31.0999755859375, 31.210622787475586, 31.07532501220703, 30.948209762573242, 30.826629638671875, 30.716211318969727]

[32.948448181152344, 33.099937438964844, 33.23990249633789, 33.43804931640625, 33.688114166259766, 34.05646896362305, 33.5637092590332, 34.05782699584961, 33.69127655029297, 33.45049285888672, 33.257877349853516, 33.09467697143555, 32.953392028808594]
[31.496746063232422, 31.628128051757812, 31.76934242248535, 31.8779354095459, 32.11865997314453, 32.346561431884766, 0.0, 32.336822509765625, 32.0954704284668, 31.918598175048828, 31.77651023864746, 31.641735076904297, 31.513755798339844]
[31.346698760986328, 31.46917724609375, 31.57929801940918, 31.65304183959961, 31.814544677734375, 31.50114631652832, 0.0, 31.472368240356445, 31.788970947265625, 31.691736221313477, 31.580381393432617, 31.47230339050293, 31.364946365356445]

WP_PSNR = []
MC_PSNR = []
RE_PSNR = []

        
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
ytick_list = [range(29,32),range(31,34),range(32,35),range(33,36)]
for i in range(4):
    PSNRs = [LSVC_error[i],DVC_error[i],RLVC_error[i]]
    ylabel = 'PSNR (dB)' if i==0 else ''
    legloc = 'lower center' if i==0 else 'best'
    line_plot(frame_loc,PSNRs,eplabels,colors,
            f'/home/bo/Dropbox/Research/SIGCOMM22/images/error_prop_{i}.eps',
            'Frame Index',ylabel,xticks=range(1,14),yticks=ytick_list[i],
            lfsize=lfsize,legloc=legloc)

Ubpps = [[0.12,0.18,0.266,0.37,0.50],
		[0.12,0.20,0.33,0.54],
		[0.14,0.24,0.40,0.67],
		[0.08,0.12,0.19,0.27],
		[0.06,0.11,0.164,0.24],
		]
UPSNRs = [[30.63,32.17,33.52,34.39,35.01],
		[30.58,32.26,33.75,34.97],
		[31.53,33.05,34.33,35.36],
		[29.52,31.30,32.52,33.28],
		[29.42,31.30,32.60,33.42],
		]
Ubpps = np.array(Ubpps)
UPSNRs = np.array(UPSNRs)
line_plot(Ubpps,UPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-UVG.eps',
		'bpp','PSNR (dB)',use_arrow=True,arrow_coord=(0.1,34),
		xticks=[.2,.4,.6],yticks=range(30,37))

line_plot(Ubpps[1:],UPSNRs[1:],labels[1:],colors[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation0.eps',
		'bpp','PSNR (dB)',use_arrow=True,arrow_coord=(0.1,34),
		xticks=[.2,.4,.6],yticks=range(30,37))

Mbpps = [[0.14,0.21,0.30,0.41,0.538],
		[0.14,0.23,0.38,0.63],
		[0.16,0.26,0.43,0.76],
		[0.09,0.15,0.22,0.31],
		[0.08,0.14,0.20,0.28],
		]
MPSNRs = [[30.93,32.47,33.75,34.57,35.18],
		[30.71,32.42,33.95,35.23],
		[31.56,33.16,34.52,35.61],
		[29.98,31.72,32.96,33.73],
		[29.64,31.54,32.80,33.60],
		]

line_plot(Mbpps,MPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-MCL.eps',
		'bpp','',
		xticks=[.2,.4,.6],yticks=range(30,37))

Xbpps = [[0.10,0.147,0.22,0.34,0.489],
		[0.10,0.17,0.32,0.60],
		[0.11,0.21,0.37,0.66],
		[0.06,0.10,0.16,0.24],
		[0.05,0.087,0.138,0.216],
		]
XPSNRs = [[31.84,33.24,34.51,35.40,36.08],
		[31.56,32.97,34.30,35.49],
		[32.45,33.83,35.01,35.96],
		[30.89,32.68,33.92,34.69],
		[30.75,32.51,33.71,34.48],
		]

line_plot(Xbpps,XPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-Xiph.eps',
		'bpp','',
		xticks=[.2,.4,.6],yticks=range(31,37))

Hbpps = [[0.14,0.21,0.307,0.37,0.56],
		[0.14,0.24,0.40,0.66],
		[0.16,0.275,0.47,0.77],
		[0.09,0.15,0.22,0.32],
		[0.08,0.13,0.20,0.28],
		]
HPSNRs = [[29.75,31.28,32.39,33.05,33.54],
		[29.56,31.18,32.66,33.86],
		[30.50,32.01,33.28,34.24],
		[28.84,30.40,31.35,31.95],
		[28.48,30.29,31.37,32.04],
		]

line_plot(Hbpps,HPSNRs,labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/rate-distortion-HEVC.eps',
		'bpp','',
		xticks=[.2,.4,.6,.8],yticks=range(30,35))

def groupedbar(data_mean,data_std,ylabel,path,yticks=None,envs = ['WiFi', 'Lossy WiFi'],
				methods = ['LSVC','H.264','H.265','DVC','RLVC'],use_barlabel=False):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)
	colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']

	ax.grid()
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['top'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	plt.xticks(np.linspace(1, 2, 2), envs, size=labelsize)
	ax.set_ylabel(ylabel, size=labelsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
	hatches = ['/', '-', 'O', '|', '\\']

	for i in range(num_methods):
	    x_index = center_index + (i - (num_methods - 1) / 2) * 0.16
	    hbar=plt.bar(x_index, data_mean[:, i], width=0.15, linewidth=2,
	            color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
	    plt.errorbar(x=x_index, y=data_mean[:, i],
	                 yerr=data_std[:, i], fmt='k.', elinewidth=3)
	    if use_barlabel:
	    	ax.bar_label(hbar, fmt='%.2f', fontsize = labelsize_b-16, rotation='vertical')
	

	plt.legend(bbox_to_anchor=(0.46, 1.28), fancybox=True,
	           loc='upper center', ncol=3, fontsize=20)
	fig.savefig(path, bbox_inches='tight')
	plt.close()


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

hbar_plot(fps_avg_list[1:],fps_std_list[1:],labels[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation1.eps',
		'#4f646f','FPS')

with open('cpu_speed.log','r') as f:
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

fps_avg_arr = np.array(fps_avg_list)
fps_std_arr = np.array(fps_std_list)
fps_avg_arr.resize(2,5)
fps_std_arr.resize(2,5)
selected = [0,3,4]
groupedbar(fps_avg_arr[:,selected],fps_std_arr[:,selected],'FPS', 
	'/home/bo/Dropbox/Research/SIGCOMM22/images/speed.eps',yticks=[10,20,30],
	envs = ['GTX 1080', 'Intel i9'], methods = ['LSVC','DVC','RLVC'],use_barlabel=True)


# NET 1
ytick_list = [range(30,36),range(30,36),range(31,36),range(29,34)]
fps_arr = get_arr_from(3,'live_client.log')
fps_arr = np.mean(fps_arr,2)

k = 0
for psnr,bpp,yc in [(UPSNRs,Ubpps,True),(MPSNRs,Mbpps,False),(XPSNRs,Xbpps,False),(HPSNRs,Hbpps,False)]:
	throughput = np.array(bpp)/fps_arr
	# used to compute throughput
	line_plot(throughput,psnr,labels,colors,
		f'/home/bo/Dropbox/Research/SIGCOMM22/images/bpep-distortion_{k}.eps',
		'Pixel-level Throughput','PSNR (dB)' if yc else '',use_arrow=yc,arrow_coord=(0.005,34),
		xticks=[.01,.02],yticks=ytick_list[k])
	k += 1

fps_avg1,fps_std1 = get_mean_std_from(3,'live_client.log')
rbf_avg1,rbf_std1 = get_mean_std_from(4,'live_server.log')
lat_avg1,lat_std1 = get_mean_std_from(5,'live_server.log')

# NET 2
fps_arr = get_arr_from(3,'lossy_client.log')
fps_arr = np.mean(fps_arr,2)

k = 0
for psnr,bpp,yc in [(UPSNRs,Ubpps,True),(MPSNRs,Mbpps,False),(XPSNRs,Xbpps,False),(HPSNRs,Hbpps,False)]:
	throughput = np.array(bpp)/fps_arr
	# used to compute throughput
	line_plot(throughput,psnr,labels,colors,
		f'/home/bo/Dropbox/Research/SIGCOMM22/images/bpep-distortion_lossy_{k}.eps',
		'Pixel-level Throughput','PSNR (dB)' if yc else '',use_arrow=yc,arrow_coord=(0.005,34),
		xticks=[.01,.02],yticks=ytick_list[k])
	k += 1

fps_avg2,fps_std2 = get_mean_std_from(3,'lossy_client.log')
rbf_avg2,rbf_std2 = get_mean_std_from(4,'lossy_server.log')
lat_avg2,lat_std2 = get_mean_std_from(5,'lossy_server.log')

fps_avg = np.stack((fps_avg1,fps_avg2))
fps_std = np.stack((fps_std1,fps_std2))
rbf_avg = np.stack((rbf_avg1,rbf_avg2))
rbf_std = np.stack((rbf_std1,rbf_std2))
lat_avg = np.stack((lat_avg1,lat_avg2))
lat_std = np.stack((lat_std1,lat_std2))

groupedbar(fps_avg,fps_std,'FPS', 
	'/home/bo/Dropbox/Research/SIGCOMM22/images/framerate.eps',
	yticks=[0,10,20,30])
groupedbar(rbf_avg,rbf_std,'', 
	'/home/bo/Dropbox/Research/SIGCOMM22/images/rebufferrate.eps',
	yticks=[0,.1,.2,.3])
groupedbar(lat_avg,lat_std,'Second', 
	'/home/bo/Dropbox/Research/SIGCOMM22/images/latency.eps',
	yticks=[0,1,2,3,4])
########################ABLATION####################################
# UVG
ab_labels = ['LSVC','w/o TSE','Linear','One-hop']
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
		'/home/bo/Dropbox/Research/SIGCOMM22/images/ablation_e.eps',
		'bpp','PSNR (dB)',use_arrow=True,arrow_coord=(0.15,33),
		xticks=[.1,.2,.3,.4],yticks=range(30,35))

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
		'#4f646f','FPS',yticks=range(0,40,10))

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
GOP_size = [[(i+1)*2+1 for i in show_indices] for _ in range(len(scalability_labels))]
line_plot(GOP_size,fps_avg_list[:,show_indices],scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_fps.eps',
		'GOP Size','FPS',yerr=fps_std_list[:,show_indices],ncol=0,
		yticks=range(10,50,10),xticks=range(0,61,10))
line_plot(GOP_size,gpu_avg_list[:,show_indices],scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_gpu.eps',
		'GOP Size','GPU Usage (%)',xticks=range(0,61,10),yticks=[.2,.3,.4,.5],
		yerr=gpu_std_list[:,show_indices])

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
		'/home/bo/Dropbox/Research/SIGCOMM22/images/scalability_e.eps',
		'bpp','PSNR (dB)',use_arrow=True,arrow_coord=(0.15,34),
		xticks=[0.1,0.2,.3,.4,.5,.6,.7],yticks=range(30,35))

# motiv
show_indices = range(30)
GOP_size = [[(i+1)*2+1 for i in show_indices] for _ in range(2)]
line_plot(GOP_size,gpu_avg_list[1:,show_indices],scalability_labels[1:],colors[1:],
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation2.eps',
		'GOP Size','GPU Usage (%)',xticks=range(0,61,10),yticks=[.21,.22,.23,.24])
show_indices = range(30)#[0,1,5,13,29]
GOP_size = [[(i+1)*2+1 for i in show_indices] for _ in range(3)]
total_time = 1/fps_avg_list[:,show_indices]*(1+np.array(show_indices))
line_plot(GOP_size,total_time,scalability_labels,colors,
		'/home/bo/Dropbox/Research/SIGCOMM22/images/motivation3.eps',
		'GOP Size','Second',xticks=range(0,61,10),yticks=[0,.5,1,1.5])