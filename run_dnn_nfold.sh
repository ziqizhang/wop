#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src
workingdir=/home/zz/Work
#emb_format=gensim
#emb_file=/data/embeddings/glove.840B.300d.bin.gensim
emb_format=fasttext
emb_file=/data/embeddings/wop/fasttext_wop_cbow.bin
#embfile=/data/embeddings/wop/fasttext_wop_skip.bin
#for file in "$dir/"*


echo "++++++ glove n +++++"
for file in $workingdir"/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove c +++++"
for file in $workingdir"/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove n+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done


echo "++++++ glove n+c.new+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove d +++++"
for file in $workingdir"/wop/input/dnn/dnn_d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove n+d+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove n+d+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done

echo "++++++ glove n+d+c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file $workingdir $emb_format else embedding_file=$emb_file
done
