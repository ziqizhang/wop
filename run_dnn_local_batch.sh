#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src

#for file in "$dir/"*

echo "++++++ GloVe +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False 
done

echo "++++++ Name Skip +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/name_skip.bin
done

echo "++++++ Desc Skip +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin
done


