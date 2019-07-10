#!/bin/bash


export PYTHONPATH=/home/zz/Work/wop/code/python/src
workingdir=/home/zz/Work
#train=/home/zz/Work/data/Rakuten/rdc-catalog-gold-small1.tsv
#est=/home/zz/Work/data/Rakuten/rdc-catalog-gold-small2.tsv
train=/home/zz/Work/data/Rakuten/rdc-catalog-train_fasttext.tsv
test=/home/zz/Work/data/Rakuten/rdc-catalog-gold_fasttext.tsv
emb_file=none
#emb_file=/data/embeddings/wop/fasttext_wop_cbow.vec
#emb_file=/data/embeddings/wop/fasttext_wop_skip.vec

#for file in "$dir/"*


echo "++++++ glove n +++++"
for file in $workingdir"/wop/input/dnn/dnn_n/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove c +++++"
for file in $workingdir"/wop/input/dnn/dnn_c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done


echo "++++++ glove n+c.new+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove d +++++"
for file in $workingdir"/wop/input/dnn/dnn_d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+c +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done

echo "++++++ glove n+d+c.new +++++"
for file in $workingdir"/wop/input/dnn/dnn_n+d+c.new/"*.txt
do
     echo "File is '$file'"
     python3 -m exp.rakuten.exp_rakuten_scalable $file $workingdir $train $test gensim fasttext class_column=1 training_text_data_columns=0,name,20 embedding_file=$emb_file
done
