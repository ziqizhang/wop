'''
given a product name, measure the similarity of its words against 'consumer' and 'product'. and select only those highly similar words

'''

import sys,csv

from util import nlp
from sematch.semantic.similarity import WordNetSimilarity
import operator
import pandas as pd

target_words=["product","consumer"]
sim_ww_cache={}
sim_iw_cache={}
wns = WordNetSimilarity()

def select_similar(input_words:set, target_words:list,
                   topN:int, wns_method):
    score_map={}
    for iw in input_words:
        if iw in sim_iw_cache.keys():
            score_map[iw] = sim_iw_cache[iw]
            continue

        isum=0
        for tw in target_words:
            key=iw+tw
            if key in sim_ww_cache.keys():
                score=sim_ww_cache[key]
            else:
                #if wns_method=='lin':
                score=wns.word_similarity(iw, tw,'lin')
                sim_ww_cache[key]=score
            isum+=score
        isum=isum/len(target_words)
        score_map[iw]=isum
        sim_iw_cache[iw]=isum

    sorted_keys = sorted(score_map.items(), key=operator.itemgetter(1), reverse=True)

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

if __name__ == "__main__":
    #load dataset
    in_data_csv=sys.argv[1]
    out_data_csv=sys.argv[2]
    name_col=4
    replace_content_col=5
    topN=3

    #take name and expand
    df = pd.read_csv(in_data_csv, header=0, delimiter=';', quoting=0, encoding="utf-8")
    headers = list(df.columns.values)

    df = df.as_matrix()
    with open(out_data_csv, 'w',newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(headers)
        for row in df:
            name=row[name_col]
            input_words=select_input_words(name)
            similar=select_similar(input_words, target_words,topN, "lon")

            line=" ".join(similar)
            row[replace_content_col]=line.strip()
            csvwriter.writerow(row)

    #save