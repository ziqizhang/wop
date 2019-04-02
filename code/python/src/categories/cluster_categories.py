import functools

import datetime
import gensim
import numpy
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import csv

from sklearn.metrics import silhouette_score, calinski_harabaz_score

from util import nlp
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

word_vectorizer = TfidfVectorizer(
    # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    preprocessor=nlp.normalize,
    tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=2),
    ngram_range=(1, 1),
    use_idf=True,
    smooth_idf=False,
    norm=None,  # Applies l2 norm smoothing
    decode_error='replace',
    max_features=50000,
    min_df=1,
    max_df=0.99
)

def create_text_input_data(col_str:str, df):
    cols=col_str.split("-")
    if (len(cols)==1):
        text_data= df[:, int(col_str)]
        text_data = ["" if (x is numpy.nan or x == 'nan') else x for x in text_data]
        return text_data
    text_cols=[]
    for c in cols:
        text_data = df[:, int(c)]
        text_data = ["" if (x is numpy.nan or x == 'nan') else x for x in text_data]
        text_cols.append(list(text_data))
        list_of_separators = [' ' for i in range(len(text_data))]
        text_cols.append(list_of_separators)
    texts= numpy.stack(text_cols, axis=-1)

    df= pd.DataFrame(texts)
    texts_final=None
    for column in df:
        if texts_final is None:
            texts_final=df[column]
        else:
            texts_final=texts_final.astype(str)+df[column].astype(str)
    return texts_final

#read category/breadscrum data from the input dataset.
#return a dictionary indexing of category string and id
#cat_col: should be a string of col indexes separated by '-'
def read_categories(in_file:str, cat_col:str):
    df = pd.read_csv(in_file, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     ).as_matrix()
    cat_texts = create_text_input_data(cat_col, df)

    # cat_texts_raw=create_text_input_data(cat_col,df)
    # cat_texts=[]
    # for c in cat_texts_raw:
    #     if len(c.split(" "))<5:
    #         cat_texts.append(c)

    idx=0
    cat_to_idx=dict()
    idx_to_cat=dict()
    for cat in cat_texts:
        if cat in cat_to_idx.keys():
            continue
        else:
            idx+=1
            cat_to_idx[cat]=idx
            idx_to_cat[idx]=cat

    return cat_to_idx, idx_to_cat, cat_texts

#calculate tfidf of vocab in categories
#return dictionary of tfidf scores
def calc_tfidf(cats:list):
    res = word_vectorizer.fit_transform(cats)
    tfidf=res.toarray()
    tfidf = min_max_scaler.fit_transform(tfidf)
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    inverted_vocab = dict([[v, k] for k, v in vocab.items()])
    return tfidf, inverted_vocab

#def create feature space for clustering
#return X the feature matrix for clustering
def represent_categories(tfidf_scores:numpy.ndarray, cat_vocab:dict, embedding_model_file:str, embedding_dim:int):
    gensimFormat = ".gensim" in embedding_model_file
    if gensimFormat:
        model = gensim.models.KeyedVectors.load(embedding_model_file, mmap='r')
    else:
        model = gensim.models.KeyedVectors. \
            load_word2vec_format(embedding_model_file, binary=True)

    cat_matrix=numpy.ndarray(shape=(len(tfidf_scores),embedding_dim), dtype=float)
    count=0
    for prod_cat in tfidf_scores:
        print(count)
        #retrieve words in this cat_str and their tfidf weights
        non_zero_idx=[i for i, e in enumerate(prod_cat) if e != 0]

        #find embedding
        word_embeddings=[]
        for word_idx in non_zero_idx:
            word=cat_vocab[word_idx]
            if word in model.wv.vocab.keys():
                vec = model.wv[word]
                tfidf=prod_cat[word_idx]
                #tfidf=1.0
                vec = numpy.array(vec, dtype=float) * tfidf
                word_embeddings.append(vec)

        #aggregate embedding
        if len(word_embeddings)>0:
            cat_vector=numpy.sum(word_embeddings, axis=0)
        else:
            cat_vector=numpy.zeros(embedding_dim)
        # try:
        #     print("\t"+str(len(cat_vector)))
        # except TypeError:
        #     print("wrong")
        cat_matrix[count]=cat_vector
        count += 1

    return cat_matrix


#returns cluster membership of categories
def cluster(X, n_clusters):
    if type(n_clusters) is list:
        print("optmising cluster size for "+str(len(n_clusters))+" values")

        max_opt_score=0.0
        best_n=0
        best_clustering=None
        for n in n_clusters:
            print("\t"+str(datetime.datetime.now()))
            print("\tclustering with n="+str(n))
            clustering = AgglomerativeClustering(n_clusters=n).fit(X)
            cluster_labels=clustering.labels_
            #opt_score = silhouette_score(X, cluster_labels)
            opt_score = calinski_harabaz_score(X, cluster_labels)
            if opt_score>max_opt_score:
                max_opt_score=opt_score
                best_n=n
                best_clustering=cluster_labels
            print("\tsilhouette avg="+str(opt_score))
        print("best cluster size is "+str(best_n))
        return best_clustering

    else:
        print("creating "+str(n_clusters)+" clusters")
        print("\t" + str(datetime.datetime.now()))
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        cluster_labels = clustering.labels_
        return cluster_labels



#insert category cluster membership into original data file
def write_category_membership(in_file:str, cat_cluster:list, out_file_name):
    df = pd.read_csv(in_file, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     )
    header=list(df.columns.values)
    df=df.as_matrix()

    with open(out_file_name, mode='w') as employee_file:
        csv_writer = csv.writer(employee_file, delimiter=';', quotechar='"', quoting=0)
        csv_writer.writerow(header)
        for i in range(len(df)):
            row=df[i]
            row=numpy.append(row, cat_cluster[i])
            csv_writer.writerow(row)


if __name__ == "__main__":
    cat_to_idx, idx_to_cat, cat_texts=read_categories(sys.argv[1],sys.argv[2])
    print("total unique categories="+str(len(set(cat_texts))))
    tfidf, cat_vocab=calc_tfidf(cat_texts)
    cat_matrix=represent_categories(tfidf,cat_vocab, sys.argv[3],300)

    cluster_labels=None
    if len(sys.argv)>4:
        n_cluster=int(sys.argv[4])
        cluster_labels =cluster(cat_matrix, n_cluster)
    else:
        n_clusters=[10,25,50,150,200,250,500,1000]
        cluster_labels =cluster(cat_matrix, n_clusters)

    out_file=sys.argv[1][0:len(sys.argv[1])-4]+"_cat_cluster.csv"

    write_category_membership(sys.argv[1], cluster_labels, out_file)