#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src
workingdir=/home/zz/Work
#train=/home/zz/Work/data/Rakuten/rdc-catalog-gold-small1.tsv
#test=/home/zz/Work/data/Rakuten/rdc-catalog-gold-small2.tsv
train=/home/zz/Work/data/Rakuten/rdc-catalog-train_fasttext.tsv
test=/home/zz/Work/data/Rakuten/rdc-catalog-gold_fasttext.tsv
emb_format=gensim
emb_file=/data/embeddings/glove.840B.300d.bin.gensim
#emb_format=fasttext
#emb_file=/data/embeddings/wop/fasttext_wop_cbow.bin
#emb_file=/data/embeddings/wop/fasttext_wop_skip.bin

#for file in "$dir/"*


echo "++++++ glove n +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove c.new +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done


echo "++++++ glove n+c.new+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove d +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+c +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+d+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+c.new +++++"
for file in "/home/zz/Work/wop/input/dnn/dnn_n+d+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test $emb_format class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done
