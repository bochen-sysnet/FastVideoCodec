#!/bin/bash

# echo "Local encoder test"

# python eval.py --task x264

# python eval.py --task x265

# python eval.py --task DVC --encoder_test

# python eval.py --task RLVC --encoder_test

# python eval.py --task SPVC --encoder_test

##############################################

echo "No loss remote test on 130.126.136.154"
ssh monet@130.126.136.154 "/home/monet/anaconda3/bin/activate yolov5;cd research/FastVideoCodec;python eval.py --task x264 --role client" &
python eval.py --task x264 --role server
  
# python eval.py --task x265 --role client &
# python eval.py --task x265 --role server

# python eval.py --task RLVC --role client &
# python eval.py --task RLVC --role server
  
# python eval.py --task DVC --role client &
# python eval.py --task DVC --role server

# python eval.py --task SPVC --role client &
# python eval.py --task SPVC --role server

echo "Loss: 10%"
# sudo tc qdisc add dev lo root netem loss 10%

# sudo tc qdisc del dev lo root

##############################################

echo "No loss remote test on 10.251.114.121"