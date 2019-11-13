import functools

import numpy
import pandas as pd
import re
import csv
import operator

from sklearn.feature_extraction.text import CountVectorizer

from util import nlp

word_vectorizer = CountVectorizer(
    # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    preprocessor=nlp.normalize,
    tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=1),
    ngram_range=(1, 1),
    stop_words=nlp.stopwords,  # We do better when we keep stopwords
    decode_error='replace',
    max_features=50000
)

def output_names(in_data_csv, separator, name_col, out_file):
    df = pd.read_csv(in_data_csv, header=0, delimiter=separator, quoting=0, encoding="utf-8",
                     ).as_matrix()
    wf = open(out_file, "w")
    for r in df:
        name=r[name_col]
        name=re.sub('[^0-9a-zA-Z\-]+', ' ', name)
        name=re.sub(' +', ' ', name)
        wf.write(name+"\n")

def replace_desc(in_data_csv, separator, replace_col, in_replace_data_file, out_file,
                 keywords_only=True):
    df = pd.read_csv(in_data_csv, header=0, delimiter=separator, quoting=0, encoding="utf-8",
                     )
    headers=list(df.columns.values)

    df=df.as_matrix()
    f= open(in_replace_data_file, 'r')
    replace_data = f.readlines()

    with open(out_file, 'w',newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=separator,
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(headers)

        for i in range(len(df)):
            r = df[i]
            replace_data_line=replace_data[i]
            if keywords_only:
                r[replace_col]=extract_keywords_from_desc(replace_data_line.strip())
            else:
                r[replace_col] = replace_data_line.strip()
            nr = []
            for v in r:
                if type(v) is float and numpy.isnan(v):
                    v=""
                nr.append(str(v))
            csvwriter.writerow(nr)

def extract_keywords_from_desc(desc, num_of_words=5):
    #print("now="+desc)
    if len(desc)==0:
        return ""
    freq = word_vectorizer.fit_transform([desc]).toarray()
    freq=freq[0]
    freq_map={}
    for i in range(len(freq)):
        freq_map[i]=freq[i]
    sorted_freq = sorted(freq_map.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    vocab = {v: k for k, v in vocab.items()}

    str=""

    collected=0
    for i in range(len(sorted_freq)):
        if collected>=num_of_words:
            break
        k = sorted_freq[i][0]
        v = vocab[k]
        if len(v)==1 and collected>0:
            collected=collected-1
            continue
        collected+=1
        str+=v+" "


    return str.strip()

if __name__ == "__main__":

    #extract names from original input file and output as a list
    # in_data_csv="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv"
    # name_col=4
    # out_file="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_names.txt"
    # output_names(in_data_csv, ';',name_col, out_file)

    #merge new descriptions with origianl input csv file
    in_data_csv="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv"
    separator=';'
    replace_col=5
    in_replace_data_file="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_descexp.txt"
    out_file="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_descexp.csv"
    replace_desc(in_data_csv,separator,replace_col,in_replace_data_file,out_file, keywords_only=True)