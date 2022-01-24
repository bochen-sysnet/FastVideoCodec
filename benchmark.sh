#!/bin/bash

# impact of network: rebuffer,fps,start-up
for test_num in 1 2 3 #4 5 
do
	echo "Role:$1. IP:$2. Test: $test_num. Loss: $lr"	
	for task in SPVC64-N DVC RLVC x264 x265
	do
		python3 eval.py --task $task --role $1 --server_ip $2 --client_ip $3
		# sudo kill -9 `sudo lsof -t -i:8846`
	done
done
# sudo tc qdisc add dev wlp68s0 root netem loss 20%
# sudo tc qdisc del dev wlp68s0 root

# -----------------------------------------------------------
# live
# rebuffer,fps
# for dataset in UVG MCL-JCV Xiph HEVC
# do
# 	for test_num in 1 2 3 #4 5 
# 	do
# 		echo "Role:$1. IP:$2. Test: $test_num. Data: $dataset"	
# 		for task in SPVC64-N DVC RLVC x264 x265
# 		do
# 			python3 eval.py --task $task --role $1 --server_ip $2 --dataset $dataset
# 			# sudo kill -9 `sudo lsof -t -i:8846`
# 		done
# 	done
# done

# start-up can be tested in standalone

# offline
# efficiency: on-going now
# speed
# for task in SPVC64-N
# do
# 	python3 eval.py --task $task --encoder_test --Q_option Slow
# done
# # python eval.py --task x265,x264 --fps 1000
# for task in x264 x265
# do
# 	python3 eval.py --task $task --fps 1000 --Q_option Slow
# done

# -----------------------------------------------------------
# impact of hardware
# python eval.py --task SPVC64-N,DVC,RLVC --encoder_test --Q_option Slow
# take the smaller one into account
# python eval.py --task x265,x264 --fps 1000 --Q_option Slow

# -----------------------------------------------------------
# eval scalability: use different models measure mean,std on UVG
# dynamic
# for task in RLVC
# do
# 	for p_num in 3 4 5 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29  #1 2 6 14 30 # 
# 	do
# 		python3 eval.py --task $task --encoder_test --fP $p_num --bP $p_num --Q_option Slow
# 	done
# done
# python eval.py --task SPVC64-N,DVC,RLVC --encoder_test --fP 1,2,6,14,30 --bP 0
# static 
# python eval.py --task LSVC-A,DVC-pretrained,RLVC2 --mode static --fP 1,2,6,14,30..

# -----------------------------------------------------------
# error propagation
# python eval.py --task LSVC-A,DVC-pretrained,RLVC2 --mode static
# watch result

# -----------------------------------------------------------
# ablation
# efficiency:trivial
# speed
# python eval.py --task SPVC64-N,SPVC64-N-D,SPVC64-N-L,SPVC64-N-O,SPVC64-N-P, --encoder_test