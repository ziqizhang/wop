#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src

#for file in "$dir/"*

for file in "/home/zz/Work/wop/input/dnn_name+catcluster/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False
done


