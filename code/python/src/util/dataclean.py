import csv
import urllib.request

import numpy as np
import pandas as pd
import glob
import os

def optimize_solr_index(solr_url, corename):
    code = urllib.request. \
        urlopen("{}/{}/update?optimize=true".
                format(solr_url, corename)).read()


def merge_csv_files(csv1, csv1_id_col, csv1_label_col, csv2, csv2_id_col, outfile):
    df = pd.read_csv(csv2, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    csv2_dict = dict()
    for row in df:
        row = list(row)
        id = row[csv2_id_col]
        del row[csv2_id_col]
        csv2_dict[id] = row

    df = pd.read_csv(csv1, header=None, delimiter=",", quoting=0
                     ).as_matrix()
    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in df:
            row = list(row)
            id = row[csv1_id_col]
            label = row[csv1_label_col].strip()
            del row[csv1_label_col]
            row = row + csv2_dict[id] + [label]
            csvwriter.writerow(row)


#an ad hoc method to remove the """ chars in csv files
def clean_data(folder):
    for file in os.listdir(folder):
        df = pd.read_csv(folder+"/"+file,
                         header=0, delimiter=",", quoting=0, quotechar='"')
        header = list(df.columns.values)
        df = df.as_matrix()
        with open(folder+"/"+file, 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(header)
            for row in df:
                for i in range(0, len(row)):
                    if type(row[i]) is str and row[i].endswith('"'):
                        row[i]=row[i][0:len(row[i])-1]
                csvwriter.writerow(row)


def replace_nan_in_list(data):
    for i in range(0, len(data)):
        if type(data[i]) is not str and np.isnan(data[i]):
            print("\t\t data row "+str(i)+" is empty")
            data[i]=" "
    return data

