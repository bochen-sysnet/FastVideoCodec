#!/bin/bash

# impact of network: rebuffer,fps,start-up
# sudo tc qdisc add dev wlp68s0 root netem loss 20%
# for test_num in 1 2 3 
# do
# 	echo "Role:$1. IP:$2. Test: $test_num. Loss: $lr"	
# 	for task in SPVC64-N x264 x265 DVC RLVC
# 	do
# 		python3 eval.py --Q_option Slow --task $task --role $1 --server_ip 130.126.136.154 --client_ip 10.194.246.197
# 		# sudo kill -9 `sudo lsof -t -i:8846`
# 	done
# done
# sudo tc qdisc del dev wlp68s0 root

# -----------------------------------------------------------
# live
# rebuffer,fps
# for task in SPVC64-N x264 x265 DVC RLVC
# do
# 	python3 eval.py --Q_option Slow --task $task --role $1 --server_ip 130.126.136.154 --client_ip 10.194.246.197 --dataset $dataset
# done

# offline
# efficiency: 
# speed
# for task in SPVC64-N
# do
# 	python3 eval.py --task $task --encoder_test --Q_option Slow
# done

# -----------------------------------------------------------
# impact of hardware
# for task in SPVC64-N DVC RLVC
# do
# 	python3 eval.py --Q_option Slow --task $task --encoder_test --no-use_cuda
# done
# python eval.py --task x264 --fps 1000 --Q_option Slow
# python eval.py --task x265 --fps 1000 --Q_option Slow

# -----------------------------------------------------------
# eval scalability: use different models measure mean,std on UVG
# dynamic
# for task in SPVC64-N
# do
# 	for p_num in {1..14} 
# 	do
# 		python3 eval.py --task $task --encoder_test --fP $p_num --bP $p_num --Q_option Slow
# 	done
# done

# -----------------------------------------------------------
# error propagation
# python eval.py --task LSVC-A,DVC-pretrained,RLVC2 --mode static
# watch result

# -----------------------------------------------------------
# ablation
# efficiency:trivial
# for task in SPVC64-N-O
# do
#	python3 eval.py --task $task --encoder_test --Q_option Slow
# done

for test_num in 1 2 3 4 5
do
	for task in RLVC DVC SPVC
	do
		python3 eval.py --task $task --encoder_test --role client
		python3 eval.py --task $task 
	done
done