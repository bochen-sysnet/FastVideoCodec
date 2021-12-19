#!/bin/bash
 
for lr in 0% 10%
do
	if [ $1 == 'client' ]
	then
	    echo "Enable: $lr loss"
	fi
	for test_num in 1 2
	do
		echo "Role:$1. IP:$2. Test: $test_num. Loss: $lr"	
		# python eval.py --task x264 --role $1 --server_ip $2 
		# python eval.py --task x265 --role $1 --server_ip $2
		# python eval.py --task RLVC --role $1 --server_ip $2
		# python eval.py --task  DVC --role $1 --server_ip $2
		# python eval.py --task SPVC --role $1 --server_ip $2
	done
	if [ $1 == 'client' ]
	then
	    echo "Disable: $lr loss"
	fi
done

# sudo tc qdisc add dev lo root netem loss 10%

# sudo tc qdisc del dev lo root
