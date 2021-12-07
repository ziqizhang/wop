'''
This class is used to prepare and process input/output to the openmt library that is used to map product names to
categories

OpenMT in: /home/zz/Programs/OpenNMT-py
Command: python translate.py -model /home/zz/Work/data/mt/product/model/cat_label_words_nmt_model_step_400000.pt -src /home/zz/Work/data/mt/product/translation_in/goldstandard_eng_v1_utf8_names.txt -output /home/zz/Work/data/mt/product/translation_out/goldstandard_eng_v1_utf8_names_tlw.txt -replace_unk


There are methods that read different input prod classification data and output just a list of product names

There are methods that read the MT output that is applied to the above productnames, and merge them with the
corresponding original datasets and output them
'''
from exp import exp_util
from util import nlp
import csv, re, json
import pandas as pd

def normalize_name(name):
    n = nlp.normalize(name).lower()
    n=re.sub("([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*)","LETTERNUMBER",n)
    n=re.sub("(?<!\S)\d+(?!\S)","NUMBER",n)
    return n

#method for extracting just the name columns
def extract_prodnames(inTrainFile, inTestFile, dataset_type, name_col, outFolder):
    print("loading dataset...")
    if dataset_type == "mwpd":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(inTrainFile, inTestFile)
    elif dataset_type == "rakuten":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(inTrainFile, inTestFile, delimiter="\t")
    elif dataset_type == "icecat":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonIceCAT(inTrainFile, inTestFile)
    else:  # wdc
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(inTrainFile, inTestFile)

    df1=df[0:train_size]
    df2=df[train_size:]

    filename = inTrainFile[inTrainFile.rindex("/")+1:]
    with open(outFolder+"/"+dataset_type+"_"+filename+".name", mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in df1:
            name =row[name_col]
            name=normalize_name(name).strip()
            if len(name)==0:
                name="NONE"
            csvwriter.writerow([name])

    filename = inTestFile[inTestFile.rindex("/") + 1:]
    with open(outFolder+"/"+dataset_type+"_"+filename+".name", mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in df2:
            name = row[name_col]
            name = normalize_name(name).strip()
            if len(name) == 0:
                name = "NONE"
            csvwriter.writerow([name])

#method for merging the translated categories with original data - replacing certain attribute
def merge_replaceMWPD(inFileOriginal, inFileTransCat,
                  outFile, replace="URL"):
    f= open(inFileTransCat, 'r')
    catwords = f.readlines()
    f.close()

    writer=open(outFile,"w")

    row=0
    with open(inFileOriginal) as file:
        line = file.readline()

        while line is not None and len(line)>0:
            print(row)
            js=json.loads(line)
            js[replace]=catwords[row].strip()
            jsline=json.dumps(js)
            writer.write(jsline+"\n")
            row+=1
            line=file.readline()

    writer.close()


# method for merging the translated categories with original data - replacing certain attribute
def merge_replaceWDC(inFileOriginal, inFileTransCat,
                      outFile, replace="url"):
    f = open(inFileTransCat, 'r')
    catwords = f.readlines()
    f.close()

    writer = open(outFile, "w")

    row = 0
    with open(inFileOriginal) as file:
        line = file.readline()

        while line is not None and len(line) > 0:
            js = json.loads(line)
            js[replace] = catwords[row].strip()
            jsline = json.dumps(js)
            writer.write(jsline + "\n")
            row += 1
            line = file.readline()

    writer.close()


# method for merging the translated categories with original data - replacing certain attribute
def merge_replaceIceCat(inFileOriginal, inFileTransCat,
                      outFile, replace="Description.URL"):
    f = open(inFileTransCat, 'r', encoding="utf-es")
    catwords = f.readlines()
    f.close()

    writer = open(outFile, "w")

    row = 0
    with open(inFileOriginal) as file:
        line = file.readline()

        while line is not None and len(line) > 0:
            js = json.loads(line)
            js[replace] = catwords[row].strip()
            jsline = json.dumps(js)
            writer.write(jsline + "\n")
            row += 1
            line = file.readline()

    writer.close()

# method for merging the translated categories with original data - replacing certain attribute
def merge_addRakuten(inFileOriginal, inFileTransCat,
                      outFile, delimiter="\t"):

    f = open(inFileTransCat, 'r')
    catwords = f.readlines()
    f.close()

    writer = open(outFile, "w")

    row = 0
    with open(inFileOriginal) as file:
        line = file.readline()

        while line is not None and len(line) > 0:
            line=line.strip()
            line +=delimiter+catwords[row].strip()
            writer.write(line+"\n")
            row+=1
            line = file.readline()

    writer.close()

if __name__ == "__main__":
    #print(normalize_name("asus k31cd it049t pc 6th gen intel core i7 i7 6700 16 gb ddr4 sdram 1000 gb hdd black tower"))
    #these lines extract names from dataset only
    #mwpd
    # inTrainFile="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json"
    # inTestFile="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/validation.json"
    # inValFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json"
    # dataset_type="mwpd"
    # name_col=1
    # outFolder="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names"
    # extract_prodnames(inTrainFile, inValFile, dataset_type,name_col,outFolder)
    # extract_prodnames(inTrainFile, inTestFile, dataset_type, name_col, outFolder)
    #
    # #wdc
    # inTrainFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json"
    # inTestFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_test.json"
    # dataset_type = "wdc"
    # name_col = 1
    # outFolder = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names"
    # extract_prodnames(inTrainFile, inTestFile, dataset_type, name_col, outFolder)
    #
    # #rakuten
    # inTrainFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv"
    # inTestFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-gold.tsv"
    # dataset_type = "rakuten"
    # name_col = 0
    # outFolder = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names"
    # extract_prodnames(inTrainFile, inTestFile, dataset_type, name_col, outFolder)
    #
    # # icecat
    # inTrainFile = "/home/zz/Work/data/IceCAT/icecat_data_train.json"
    # inTestFile = "/home/zz/Work/data/IceCAT/icecat_data_test.json"
    # inValFile = "/home/zz/Work/data/IceCAT/icecat_data_validate.json"
    # dataset_type = "icecat"
    # name_col = 4
    # outFolder = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names"
    # extract_prodnames(inTrainFile, inTestFile, dataset_type, name_col, outFolder)
    # extract_prodnames(inTrainFile, inValFile, dataset_type, name_col, outFolder)


    ##################################################
    # code for merging/adding translated cat to data #
    ##################################################

    #mwpd
    # inFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json"
    # inFileTransCat="/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/mwpd_train.json.catwords"
    # outFile= "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train_mtcat.json"
    # merge_replaceMWPD(inFile, inFileTransCat, outFile)
    #
    # inFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/mwpd_test.json.catwords"
    # outFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test_mtcat.json"
    # merge_replaceMWPD(inFile, inFileTransCat, outFile)
    #
    # inFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/validation.json"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/mwpd_validation.json.catwords"
    # outFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/validation_mtcat.json"
    # merge_replaceMWPD(inFile, inFileTransCat, outFile)

    # wdc
    # inFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/wdc_train.json.catwords"
    # outFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train_mtcat.json"
    # merge_replaceWDC(inFile, inFileTransCat, outFile)
    #
    # inFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_test.json"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/wdc_test.json.catwords"
    # outFile = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_test_mtcat.json"
    # merge_replaceWDC(inFile, inFileTransCat, outFile)

    # rakuten
    # inFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train.tsv"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/rakuten_train.json.catwords"
    # outFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-train_mtcat.tsv"
    # merge_addRakuten(inFile, inFileTransCat, outFile)
    #
    # inFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-gold.tsv"
    # inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/rakuten_test.json.catwords"
    # outFile = "/home/zz/Work/data/Rakuten/original/rdc-catalog-gold_mtcat.tsv"
    # merge_addRakuten(inFile, inFileTransCat, outFile)

    # icecat
    inFile = "/home/zz/Work/data/IceCAT/icecat_data_train.json"
    inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/icecat_train.json.catwords"
    outFile = "/home/zz/Work/data/IceCAT/icecat_data_train_mtcat.json"
    merge_replaceIceCat(inFile, inFileTransCat, outFile)

    inFile = "/home/zz/Work/data/IceCAT/icecat_data_test.json"
    inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/icecat_test.json.catwords"
    outFile = "/home/zz/Work/data/IceCAT/icecat_data_test_mtcat.json"
    merge_replaceIceCat(inFile, inFileTransCat, outFile)

    inFile = "/home/zz/Work/data/IceCAT/icecat_data_validate.json"
    inFileTransCat = "/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/mt/extracted_names-catwords/icecat_validation.json.catwords"
    outFile = "/home/zz/Work/data/IceCAT/icecat_data_validate_mtcat.json"
    merge_replaceIceCat(inFile, inFileTransCat, outFile)

