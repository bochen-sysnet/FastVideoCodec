#!/bin/bash

# X264
#python3 streaming.py --task x264 --server_port 8184 --probe_port 8181
  
#python3 streaming.py --task x265 --server_port 8184 --probe_port 8181

python3 streaming.py --task RLVC --server_port 8184 --probe_port 8181
  
python3 streaming.py --task DVC --server_port 8184 --probe_port 8181