from categories import cluster_categories as cc
import sys
import pandas as pd
import csv
import numpy as np

def calc_sim(cat_to_rep:dict, name_to_rep:dict):
    scores=dict()
    assigned_cats=dict()

    cache=dict()
    count_names=0
    for name, name_rep in name_to_rep.items():
        count_names+=1
        print("\t"+str(count_names))
        if name in scores.keys():
            max=scores[name]
        else:
            max=0.0

        max_cat=None
        for cat,cat_rep in cat_to_rep.items():

            key=cat+"|"+name
            if key in cache:
                cos=cache[key]
            else:
                dot = np.dot(name_rep, cat_rep)
                norma = np.linalg.norm(name_rep)
                normb = np.linalg.norm(cat_rep)
                cos = dot / (norma * normb)
                cache[key]=cos

            if cos>max:
                max=cos
                max_cat=cat
        scores[name]=max
        assigned_cats[name]=max_cat

    return assigned_cats


if __name__=="__main__":
    cat_to_idx, idx_to_cat, cat_texts = cc.read_categories(sys.argv[1], "8-9")
    print("total unique categories=" + str(len(set(cat_texts))))
    cat_tfidf, cat_vocab = cc.calc_tfidf(cat_texts)
    cat_matrix = cc.represent_categories(cat_tfidf, cat_vocab, sys.argv[2], 300)
    cat_to_rep=dict()
    for i in range(len(cat_matrix)):
        text=cat_texts[i]
        rep=cat_matrix[i]
        cat_to_rep[text]=rep

    name_to_idx, idx_to_name, name_texts = cc.read_categories(sys.argv[1], "4")
    print("total unique names=" + str(len(set(name_texts))))
    name_tfidf, name_vocab = cc.calc_tfidf(name_texts)
    name_matrix = cc.represent_categories(name_tfidf, name_vocab, sys.argv[2], 300)
    name_to_rep=dict()
    for i in range(len(name_matrix)):
        text=name_texts[i]
        rep=name_matrix[i]
        name_to_rep[text]=rep


    cache=calc_sim(cat_to_rep, name_to_rep)
    df = pd.read_csv(sys.argv[1], header=0, delimiter=";", quoting=0, encoding="utf-8",
                     )
    header = list(df.columns.values)
    header.append("assigned_cat")
    df = df.as_matrix()

    with open(sys.argv[3], mode='w') as employee_file:
        csv_writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=0)
        csv_writer.writerow(header)
        for i in range(len(df)):
            row = df[i]
            name=row[4]

            cat=cache[name]
            if cat is None:
                cat="none"
            newrow=list(row)
            newrow.append(cat)
            csv_writer.writerow(newrow)

