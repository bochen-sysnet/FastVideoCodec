#!/bin/bash

echo "No loss test"
# python streaming.py --task x264 --role Client &
# python streaming.py --task x264 --role Server
  
# python streaming.py --task x265 --role Client &
# python streaming.py --task x265 --role Server

# python streaming.py --task RLVC --role Client &
# python streaming.py --task RLVC --role Server
  
# python streaming.py --task DVC --role Client &
# python streaming.py --task DVC --role Server

python streaming.py --task SPVC --role Client &
python streaming.py --task SPVC --role Server

# 10% packet loss
# sudo tc qdisc add dev lo root netem loss 10%

# remove loss
# sudo tc qdisc del dev lo root