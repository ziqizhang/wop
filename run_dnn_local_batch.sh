#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src

#for file in "$dir/"*

echo "++++++ GloVe +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

echo "++++++ Name Skip +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/name_skip.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

echo "++++++ Desc Skip +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done


