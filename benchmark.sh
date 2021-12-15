#!/bin/bash

# no packet loss
# why not work?
# invoke remote server
python3 streaming.py --task x264 --role Server &
# invoke local client
python3 streaming.py --task x264 --role Client &
  
# python3 streaming.py --task x265 

# python3 streaming.py --task RLVC
  
# python3 streaming.py --task DVC


# 10% packet loss
# sudo tc qdisc add dev lo root netem loss 10%

# remove loss
# sudo tc qdisc del dev lo root