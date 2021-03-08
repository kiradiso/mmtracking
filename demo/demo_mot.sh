#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='3' python demo/demo_mot.py configs/mot/centertrack/centertrack_dla34.py --input data/MOT17/train/MOT17-02-SDP/img1 --output mot.mp4 --fps 30