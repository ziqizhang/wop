#read csv data as dataframe, perform stratisfied sampling and output the required sample
import collections
import csv

import pandas as pd
from sklearn import model_selection

#prepare data to fasttext format
def to_fasttext(inCSV, textCol, classCol, outfile):
    df = pd.read_csv(inCSV, delimiter="\t", quoting=0, encoding="utf-8"
                     ).as_matrix()
    df.astype(str)

    X = df[:, textCol]
    y = df[:, classCol]
    counter = collections.Counter(y)

    single_instance = 0

    with open(outfile, mode='w') as file:
        csvwriter = csv.writer(file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X)):
            label = y[i]
            if counter[label] == 1:
                single_instance += 1
                continue
            text=X[i]

            csvwriter.writerow(["__label__"+label, text])

    print(str(single_instance) + " has only one instance and are deleted")


def subset(inCSV, textCol, classCol, outfolder, percentage):
    df = pd.read_csv(inCSV, delimiter="\t", quoting=0, encoding="utf-8"
                     ).as_matrix()
    df.astype(str)

    X=df[:, textCol]
    y = df[:, classCol]
    counter=collections.Counter(y)

    X_new=[]
    y_new=[]
    single_instance=0
    for i in range(len(X)):
        label=y[i]
        if counter[label]==1:
            single_instance+=1
        else:
            X_new.append(X[i])
            y_new.append(y[i])
    print(str(single_instance)+" has only one instance and are deleted")
    X_train, X_test, y_train, y_test, \
        indices_train, indices_test= model_selection.train_test_split(X_new, y_new, range(len(X_new)), test_size=percentage, random_state=0,
                                                                      stratify=y_new)

    filename=inCSV[inCSV.rfind("/")+1: inCSV.rfind(".tsv")]
    with open(outfolder+"/"+filename+"_"+str(percentage)+".index", 'w') as f:
        for i in indices_test:
            f.write(str(i)+"\n")

    with open(outfolder+"/"+filename+"_"+str(percentage)+".tsv", mode='w') as employee_file:
        csvwriter = csv.writer(employee_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X_test)):
            label = y_test[i]
            text=X_test[i]

            csvwriter.writerow([text, label])


if __name__ == "__main__":
    #inCSV="/home/zz/Work/data/Rakuten/rdc-catalog-train.tsv"
    # outfolder="/home/zz/Work/data/Rakuten/"
    # subset(inCSV, 0, 1, outfolder, 0.2)
    #
    # inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold.tsv"
    # outfolder = "/home/zz/Work/data/Rakuten/"
    # subset(inCSV, 0, 1, outfolder, 0.2)

    inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-train.tsv"
    outCSV="/home/zz/Work/data/Rakuten/rdc-catalog-train_fasttext.tsv"
    to_fasttext(inCSV,0,1,outCSV)

    inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold.tsv"
    outCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold_fasttext.tsv"
    to_fasttext(inCSV, 0, 1, outCSV)