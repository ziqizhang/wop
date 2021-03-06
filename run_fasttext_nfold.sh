#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src
workingdir=/home/zz/Work
embfile=none
#embfile=/data/embeddings/wop/fasttext_wop_cbow.vec
#embfile=/data/embeddings/wop/fasttext_wop_skip.vec

#for file in "$dir/"*


echo "++++++ glove n +++++"
for file in $workingdir"/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove c +++++"
for file in $workingdir"/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove n+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done


echo "++++++ glove n+c.new+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove d +++++"
for file in $workingdir"/wop/input/dnn/dnn_d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove n+d+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove n+d+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

echo "++++++ glove n+d+c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.wop.exp_wop_scalable $file /home/zz/Work none fasttext embedding_file=$embfile
done

