import pandas as pd
from nltk import PorterStemmer, WordNetLemmatizer
import numpy
from categories import cleanCategories as cc

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#0=stem; 1=lem; else=nothing
def normalise_categories(in_file_name, col, stem_or_lem):
    df = pd.read_csv(in_file_name, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     ).as_matrix()

    norm_cats=set()
    max_toks=0
    for r in df:
        c = r[col]
        if type(c) is not str and numpy.isnan(c):
            c="NONE"

        toks = len(c.split(" "))
        if toks>max_toks:
            max_toks=toks
        if stem_or_lem==0:
            c=stemmer.stem(c).strip()
            if len(c)>2:
                norm_cats.add(c)
        elif stem_or_lem==1:
            c=lemmatizer.lemmatize(c).strip()
            if len(c)>2:
                norm_cats.add(c)
        else:
            norm_cats.add(c)

    norm_cats_list=list(norm_cats)
    norm_cats_list=sorted(norm_cats_list)
    print(len(norm_cats_list))
    print(max_toks)
    for nc in norm_cats_list:
        print(nc)

def get_parent_category_level(in_file_name, col):
    df = pd.read_csv(in_file_name, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     ).as_matrix()

    norm_cats = set()
    norm_cats_list=[]
    for r in df:
        c = r[col]
        if type(c) is not str and numpy.isnan(c):
            continue
        c= cc.normaliseCategories(c)
        try:
            trim = c.index(">")
        except ValueError:
            continue

        c=c[0:trim].strip()
        norm_cats.add(c)
        norm_cats_list.append(c)

    norm_cats_unique_list=sorted(list(norm_cats))
    norm_cats=sorted(norm_cats)

    for nc in norm_cats:
        print(nc)

    print("\n\n>>>>>>>>>\n\n")
    for nc in norm_cats_unique_list:
        print(nc)




if __name__ == "__main__":
    # normalise_categories("/home/zz/Work/data/wop_data/goldstandard_eng_v1_cleanedCategories.csv",
    #     13,0)

    get_parent_category_level("/home/zz/Work/data/wop_data/goldstandard_eng_v1_utf8.csv",
                         8)