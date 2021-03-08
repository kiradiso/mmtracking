#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}     # default to 29500

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# Notes
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ -> ':' is general str
# --launcher pytorch ${@:3} -> after : is index, 3 means all str after 3 element
# e.g. foo="abcqwera", ${foo:3:5} -> qwera
# $@ can extend to position parameter list start from 1
