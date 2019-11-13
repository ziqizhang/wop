'''
given a product name, rank its words based on the product keywords extracted from the WDC dataset (see product_keyword_extractor)
and only select the top N
'''

import sys,csv

from util import nlp
import operator
import pandas as pd


def select_topN(input_words:set, target_words:list,
                   topN:int):
    score_map={}
    for iw in input_words:
        if iw in target_words:
            i = target_words.index(iw)
            score_map[iw]=i

    sorted_keys = sorted(score_map.items(), key=operator.itemgetter(1))

    res=[]
    count=0
    for k in sorted_keys:
        v = score_map[k[0]]

        res.append(k[0])
        count+=1
        if count==topN:
            break
    return res

def select_topN_voting(input_words:set, freq_name_words:list, freq_cat_words:list,
                       topN:int):
    score_map={}
    for iw in input_words:
        if iw in freq_name_words:
            i_n = freq_name_words.index(iw)
        else:
            i_n=0

        if iw in freq_cat_words:
            i_c=freq_cat_words.index(iw)
        else:
            i_c=0

        score_map[iw]=i_n+i_c

    sorted_keys = sorted(score_map.items(), key=operator.itemgetter(1))

    res=[]
    count=0
    for k in sorted_keys:
        v = score_map[k[0]]

        res.append(k[0])
        count+=1
        if count==topN:
            break
    return res


def select_input_words(sent:str):
    orig_toks = nlp.tokenize(sent, 2) #keep original
    norm_toks = nlp.tokenize(sent, 1) #lemmatize
    pos_tags = nlp.get_pos_tags(orig_toks)

    selected=set()
    for i in range(0, len(pos_tags)):
        word = orig_toks[i].lower()
        if word in nlp.stopwords or len(word) < 2:
            continue
        norm = norm_toks[i]

        tag = pos_tags[i]
        if tag in ["NN", "NNS", "NNP", "NNPS"]:
            selected.add(norm)

    return selected

def read_product_keywords(in_file:str):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    words=[]
    for index, row in df.iterrows():
        words.append(row[0])
    return words

if __name__ == "__main__":
    #load dataset
    in_data_csv=sys.argv[1]
    freq_name_file=sys.argv[2]
    freq_cat_file=sys.argv[3]
    keyword_type="vote"


    name_col=4
    replace_content_col=5
    topN=3
    out_data_csv = "/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_freq"+keyword_type+str(topN)+".csv"

    #take name and expand
    df = pd.read_csv(in_data_csv, header=0, delimiter=';', quoting=0, encoding="utf-8")
    headers = list(df.columns.values)

    freq_name_words=read_product_keywords(freq_name_file)
    freq_cat_words=read_product_keywords(freq_cat_file)

    df = df.as_matrix()
    with open(out_data_csv, 'w',newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(headers)
        for row in df:
            name=row[name_col]
            input_words=select_input_words(name)

            if keyword_type=="name":
                similar=select_topN(input_words, freq_name_words, topN)
            elif keyword_type=="cat":
                similar = select_topN(input_words, freq_cat_words, topN)
            else:
                similar = select_topN_voting(input_words, freq_name_words, freq_cat_words, topN)

            line=" ".join(similar)
            row[replace_content_col]=line.strip()
            csvwriter.writerow(row)

    #save