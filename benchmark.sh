#!/bin/bash

# echo "Local encoder test"

# python eval.py --task x264

# python eval.py --task x265

# python eval.py --task DVC --encoder_test

# python eval.py --task RLVC --encoder_test

# python eval.py --task SPVC --encoder_test

##############################################

for i in 1 2 3 4 5
do
	echo "No loss test $i"	
	# python eval.py --task x264 --role $1 --server_ip $2 
	  
	# python eval.py --task x265 --role $1 --server_ip $2

	# python eval.py --task RLVC --role $1 --server_ip $2
	  
	# python eval.py --task  DVC --role $1 --server_ip $2

	# python eval.py --task SPVC --role $1 --server_ip $2
done
echo "Loss: 10%"
# sudo tc qdisc add dev lo root netem loss 10%

# sudo tc qdisc del dev lo root

##############################################

echo "No loss remote test"