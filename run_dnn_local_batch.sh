#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src

#for file in "$dir/"*


echo "++++++ Desc Cbow n +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_cbow.bin
done

echo "++++++ Desc skip n +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin
done

echo "++++++ Desc Cbow c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_cbow.bin
done

echo "++++++ Desc skip c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin
done

echo "++++++ Desc Cbow n+c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_cbow.bin
done

echo "++++++ Desc skip n+c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin
done


echo "++++++ Desc Cbow cnew+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_cbow.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

echo "++++++ Desc skip cnew+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

echo "++++++ Desc Cbow n+cnew+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_cbow.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

echo "++++++ Desc skip n+cnew+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_dnn_singlesetting $file /home/zz/Work False False embedding_file=/data/embeddings/wop/desc_skip.bin training_text_data=/data/wop/goldstandard_eng_v1_utf8_cat_cleaned_ZZ.csv
done

