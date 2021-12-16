#!/bin/bash

echo "No loss"
# python eval.py --task x264 --role client &
# python eval.py --task x264 --role server
  
# python eval.py --task x265 --role client &
# python eval.py --task x265 --role server

# python eval.py --task RLVC --role client &
# python eval.py --task RLVC --role server
  
# python eval.py --task DVC --role client &
# python eval.py --task DVC --role server

python eval.py --task SPVC --role client &
python eval.py --task SPVC --role server

echo "Loss: 10%"
# 10% packet loss
sudo tc qdisc add dev lo root netem loss 10%

# remove loss
sudo tc qdisc del dev lo root