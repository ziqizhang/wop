#read csv data as dataframe, perform stratisfied sampling and output the required sample
import collections
import csv
import json

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


''''
This method reads the json data file (train/val/test) in the SWC2020 mwpd format and save them as a matrix where each 
row is an instance with the following columns:
- 0: id
- 1: name
- 2: description
- 3: categorytext
- 4: url
- 5: lvl1
- 6: lvl2
- 7: lvl3
'''
def read_mwpdformat_to_matrix(in_file):
    matrix=[]
    with open(in_file) as file:
        line = file.readline()

        while line is not None and len(line)>0:
            js=json.loads(line)

            row=[js['ID'],js['Name'],js['Description'],js['CategoryText'],js['URL'],js['lvl1'],js['lvl2'],js['lvl3']]
            matrix.append(row)
            line=file.readline()
    return matrix


'''
given the input WDC prod cat GS product offers, and also the train/test set of 2155 clusters
containing product offers derived from this GS, split the GS into train/test containing product offers 
based on their cluster membership and the cluster's presence in train/test
'''
def split_wdcGS_by_traintestsplit(train_file, test_file, gs_file,
                                  outfolder):
    train_ids=set()
    test_ids=set()
    with open(train_file) as file:
        line = file.readline()

        js=json.loads(line)
        for ent in js:
            train_ids.add(ent['cluster_id'])

    with open(test_file) as file:
        line = file.readline()

        js=json.loads(line)
        for ent in js:
            test_ids.add(ent['cluster_id'])

    writer_train=open(outfolder+"/wdc_gs_train.json",'w')
    writer_test=open(outfolder+"/wds_gs_test.json",'w')
    with open(gs_file) as file:
        line = file.readline()

        js=json.loads(line)
        for ent in js:
            out_line = json.dumps(ent)
            if str(ent['cluster_id']) in train_ids:
                writer_train.write(out_line+"\n")
            else:
                writer_test.write(out_line+"\n")

    writer_train.close()
    writer_test.close()


def read_wdcgsformat_to_matrix(in_file):
    matrix=[]
    with open(in_file) as file:
        line = file.readline()

        while line is not None and len(line) > 0:
            ent = json.loads(line)

            #id, name, desc, brand, manufacturer, url, label
            # if ent['cluster_id']==12261043:
            #     print("")
            try:
                row=[ent['cluster_id'],"","","","",ent['url'],ent['categoryLabel']]
                schema_prop=ent['schema.org_properties']
                for d in schema_prop:
                    if '/name' in d.keys():
                        row[1]=d['/name'][1:-2].strip()
                    elif '/description' in d.keys():
                        row[2]= d['/description'][1:-2].strip()
                    elif '/brand' in d.keys():
                        row[3]=d['/brand'][1:-2].strip()
                    elif '/manufacturer' in d.keys():
                        row[4]=d['/manufacturer'][1:-2].strip()

                schema_prop = ent['parent_schema.org_properties']
                for d in schema_prop:
                    if row[1]=='' and '/name' in d.keys():
                        row[1]=d['/name'][1:-2].strip()
                    elif row[1]=='' and '/title' in d.keys():
                        row[1]=d['/title'][1:-2].strip()
                    elif row[2]=='' and'/description' in d.keys():
                        row[2]= d['/description'][1:-2].strip()
                    elif row[3]=='' and'/brand' in d.keys():
                        row[3]=d['/brand'][1:-2].strip()
                    elif row[4]=='' and'/manufacturer' in d.keys():
                        row[4]=d['/manufacturer'][1:-2].strip()

                matrix.append(row)
            except:
                print("Error encountered")

            line=file.readline()
            # row=[js['ID'],js['Name'],js['Description'],js['CategoryText'],js['URL'],js['lvl1'],js['lvl2'],js['lvl3']]
            # matrix.append(row)
        #    line=file.readline()
    return matrix


if __name__ == "__main__":
    #inCSV="/home/zz/Work/data/Rakuten/rdc-catalog-train.tsv"
    # outfolder="/home/zz/Work/data/Rakuten/"
    # subset(inCSV, 0, 1, outfolder, 0.2)
    #
    # inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold.tsv"
    # outfolder = "/home/zz/Work/data/Rakuten/"
    # subset(inCSV, 0, 1, outfolder, 0.2)

    # inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-train.tsv"
    # outCSV="/home/zz/Work/data/Rakuten/rdc-catalog-train_fasttext.tsv"
    # to_fasttext(inCSV,0,1,outCSV)
    #
    # inCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold.tsv"
    # outCSV = "/home/zz/Work/data/Rakuten/rdc-catalog-gold_fasttext.tsv"
    # to_fasttext(inCSV, 0, 1, outCSV)

    #categories_clusters_testing.json
    # read_wdcformat_to_matrix("/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/categories_clusters_training.json")
    # print("end")
    split_wdcGS_by_traintestsplit('/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/categories_clusters_training.json',
                                  '/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/categories_clusters_testing.json',
                                  '/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/categories_gold_standard_offers.json',
                                  '/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS')