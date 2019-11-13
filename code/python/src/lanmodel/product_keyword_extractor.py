'''
extracts useful key words from product names and categories, from the MT pair dataset
'''
import math
import pandas as pd
import operator
import sys, os
from util import nlp
import csv

name_word_freq={}
name_word_doc_freq={}
cat_word_freq={}
cat_word_doc_freq={}

#bnc_unifrqs.normal
def load_bnc_freq(in_bnc_file):
    min_f=sys.maxsize
    total_f=0

    bnc={}
    with open(in_bnc_file, "r") as ins:
        for line in ins:
            vals=line.strip().split()
            f = int(vals[0])
            bnc[vals[1]]=f
            if f<min_f:
                min_f=f
            total_f+=f
    return bnc, min_f, total_f

def process_file(in_csv_file):
    df = pd.read_csv(in_csv_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    line=0
    for index, row in df.iterrows():
        line+=1
        if line%10000==0:
            print("\t\t"+str(line))
        name=row[0]
        name_words=extract_words(name)
        update_word_freq(name_words, name_word_freq)
        update_doc_freq(name_words, name_word_doc_freq)
        cat=row[1]
        cat_words=extract_words(cat)
        update_word_freq(cat_words, cat_word_freq)
        update_doc_freq(cat_words, cat_word_doc_freq)
    return line

def rank_word(freq:dict, out_file):
    sorted_keys = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
    with open(out_file, 'w',newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for k in sorted_keys:
            w=k[0]
            f=k[1]
            row=[w,f]
            csvwriter.writerow(row)


def update_word_freq(words:list, freq:dict):
    for w in words:
        if w in freq.keys():
            freq[w]+=1
        else:
            freq[w]=1

def update_doc_freq(words:list, doc_freq:dict):
    word_sets=set(words)
    for w in word_sets:
        if w in doc_freq.keys():
            doc_freq[w]+=1
        else:
            doc_freq[w]=1


def calculate_weirdness(word_freq:dict, bnc_word_freq:dict,bnc_word_min:int, bnc_word_total:int):
    word_scores={}

    sum_word_f=sum(word_freq.values())
    for w, f in word_freq.items():
        if w in bnc_word_freq.keys():
            bnc_f=bnc_word_freq[w]
        else:
            bnc_f=bnc_word_min

        bnc_prob=bnc_f/bnc_word_total
        word_prob=f/sum_word_f
        word_scores[w]=word_prob/bnc_prob
    return word_scores

def calculate_tfidf(word_freq:dict, word_doc_freq:dict, docs:int):
    word_scores={}
    sum_word_f = sum(word_freq.values())
    
    for w, f in word_freq.items():
        doc_freq=word_doc_freq[w]
        if doc_freq<1:
            doc_freq=1

        idf=math.log(docs/doc_freq)
        tfidf=(f/sum_word_f)*idf
        word_scores[w]=tfidf
        
    return word_scores

def extract_words(line:str):
    line=str(line).replace("LETTERNUMBER","")
    line=str(line).replace("NUMBER","")
    norm_toks = nlp.tokenize(str(line), 1)

    words=[]
    for nt in norm_toks:
        word = nt.lower()
        if word in nlp.stopwords or len(word) < 3:
            continue

        words.append(word)
    return words

if __name__ == "__main__":
    bnc_freq, bnc_word_total, bnc_word_min = load_bnc_freq(sys.argv[3])
    in_data_folder = sys.argv[1]
    out_data_folder = sys.argv[2]
    
    total_products=0
    for csvf in os.listdir(in_data_folder):
        print("processing "+csvf)
        total_products+=process_file(in_data_folder+"/"+csvf)

    # rank_word_freq(name_word_freq, out_data_folder+"/product_nameword_freq.csv")
    # rank_word_freq(cat_word_freq, out_data_folder + "/product_catword_freq.csv")


    # weirdness=calculate_weirdness(name_word_freq, bnc_freq, bnc_word_total, bnc_word_min)
    # rank_word(weirdness, out_data_folder + "/product_nameword_weirdness.csv")
    # weirdness = calculate_weirdness(cat_word_freq, bnc_freq, bnc_word_total, bnc_word_min)
    # rank_word(weirdness, out_data_folder + "/product_catword_weirdness.csv")

    tfidf = calculate_tfidf(name_word_freq, name_word_doc_freq, total_products)
    rank_word(tfidf, out_data_folder + "/product_nameword_tfidf.csv")
    tfidf = calculate_tfidf(cat_word_freq, cat_word_doc_freq, total_products)
    rank_word(tfidf, out_data_folder + "/product_catword_tfidf.csv")