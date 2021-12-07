import csv
from util import data_util
import pandas as pd

'''
given a prediction file, read the right and wrong predictions and return two sets containing the indexes of the right
and wrong ones
'''
def read_correct_and_incorrect(prediction_file):
    correct={}
    incorrect={}

    index=0
    with open(prediction_file, encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csvreader:
            if row[2]=='ok':
                correct[index]=row[0]
            else:
                incorrect[index]=row[0]
            index+=1

    return correct, incorrect

'''
given a source prediction file and a target file, as well as the original gs data, find
- those correct in source but wrong in target
- those wrong in source but correct in target

then outpput two files for each of the above including the column matching the gs data schema
'''
def process(source_prediction, target_prediction, gs_file, dataset_type, outfolder):
    if dataset_type=="mwpd":
        df = data_util.read_mwpdformat_to_matrix(gs_file)
    elif dataset_type=="rakuten":
        df = pd.read_csv(gs_file, header=0, delimiter="\t", quoting=0, encoding="utf-8",
                            ).fillna('').as_matrix()
        df.astype(str)
    elif dataset_type=="icecat":
        df=data_util.read_icecatformat_to_matrix(gs_file)
    else:
        df=data_util.read_wdcgsformat_to_matrix(gs_file)

    source_correct, source_incorrect=read_correct_and_incorrect(source_prediction)
    target_correct, target_incorrect=read_correct_and_incorrect(target_prediction)

    target_correct_set=set(target_correct.keys())
    source_correct_set=set(source_correct.keys())
    target_incorrect_set=set(target_incorrect.keys())
    source_incorrect_set=set(source_incorrect.keys())

    new_correct=target_correct_set.difference(source_correct_set)
    new_incorrect=target_incorrect_set.difference(source_incorrect_set)

    new_correct=sorted(list(new_correct))
    new_incorrect=sorted(list(new_incorrect))

    f = open(outfolder + "/target_new_correct.csv", 'w', encoding='utf-8')
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
    for index in new_correct:
        instance=df[index]
        writer.writerow(instance)
    f.close()

    f = open(outfolder + "/target_new_incorrect.csv", 'w', encoding='utf-8')
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_ALL)
    for index in new_incorrect:
        instance=df[index]
        pred=target_incorrect[index]
        instance.append(str(pred))
        writer.writerow(instance)
    f.close()

    return df


if __name__ == "__main__":
    source_pred="/home/zz/Work/wop/output/classifier/predictions/predictions-dnn-setting_file=gslvl1_name|embedding=None.csv"
    target_pred="/home/zz/Work/wop/output/classifier/predictions/predictions-dnn-setting_file=gslvl1_n+u|embedding=None.csv"
    gs="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json"
    dataset_type="mwpd"
    outfolder="/home/zz/Work/wop/output/classifier"

    process(source_pred, target_pred, gs, dataset_type,outfolder)