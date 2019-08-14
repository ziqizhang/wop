import gpt_2_simple as gpt2
import sys
import pandas as pd
import csv
import re
import datetime

def encode_and_compress(inFile):
    gpt2.encode_dataset(inFile)

def fine_tune(inFile):
    model_name = "117M"
    gpt2.download_gpt2(model_name=model_name)  # model is saved into current directory under /models/117M/

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  inFile,
                  model_name=model_name,
                  steps=1000,
                  save_every=100)  # steps is max number of training steps

    gpt2.generate(sess)

def generate(inFile, outFile, start, end):
    with open(inFile) as f:
        lineList = f.readlines()

    model_name = "117M"
    sess = gpt2.start_tf_sess()

    gpt2.load_gpt2(sess)

    count=0
    with open(outFile, 'a+', newline='\n') as f:
        writer = csv.writer(f, delimiter=",", quotechar='"')
        for l in lineList:
            if count<start:
                count+=1
                continue

            if count>end:
                break

            print(str(datetime.datetime.now())+","+str(count))
            l = re.sub('[^0-9a-zA-Z]+', ' ', l).strip()
            texts = gpt2.generate(sess, return_as_list=True,
                                        temperature=1.0,
                                        nsamples=2,
                                        batch_size=2,
                                        length=200,
                                        prefix=l,
                                        include_prefix=False)
            row=[l]
            for t in texts:
                if l in t:
                    t=t[len(l):].strip()
                row.append(t)

            writer.writerow(row)
            count+=1

'''
The following code is to be used on collabotory

import csv
import datetime
import re

with open('goldstandard_eng_v1_utf8_names_casesensitive.txt') as f:
    lineList = f.readlines()

outFile='goldstandard_eng_v1_utf8_names_casesensitive-generated.csv'
with open(outFile, 'w', newline='\n') as f:
    writer = csv.writer(f, delimiter=",", quotechar='"')
    count=0
    for l in lineList:
        print(str(datetime.datetime.now())+","+str(count))
        l = re.sub('[^0-9a-zA-Z]+', ' ', l).strip()
        texts = gpt2.generate(sess, return_as_list=True,
                                        temperature=1.0,
                                        nsamples=5,
                                        batch_size=5,
                                        length=300,
                                        prefix=l,
                                        include_prefix=False)
        row=[l]
        for t in texts:
            if l in t:
                t=t[len(l):].strip()
            row.append(t)

        writer.writerow(row)
        count+=1
'''



def insert_into_data(inCSV:str, separator:str, add_to_col:int, out_translation, outCSV:str, sentences:int):
    data = pd.read_csv(inCSV, header=0, delimiter=separator, quoting=0, encoding="utf-8",
                       ).fillna('')
    headers= list(data.columns.values)
    if len(headers)<=add_to_col:
        headers.append("nlg_text")

    data=data.as_matrix()
    with open(out_translation) as f:
        lineList = f.readlines()

    with open(outCSV, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=separator, quotechar='"')
        writer.writerow(headers)
        for i in range(len(data)):
            row=list(data[i])
            translation=lineList[i].strip()

            all_sents = translation.split(".")
            sent = ""
            for i in range(sentences):
                if i>=len(all_sents):
                    break
                s = all_sents[i]
                sent+=s+". "
            sent=sent.strip()

            if len(row)>add_to_col:
                row[add_to_col]=sent
            else:
                row.append(sent)
            writer.writerow(row)

'''
/home/zz/Work/data/wdc/word2vec_corpus/name/n_1_0
/home/zz/Work/data/wdc/name/nlg.model

g
/home/zz/Work/data/mt/product/translation_in/goldstandard_eng_v1_utf8_names_casesensitive.txt
/home/zz/Work/data/mt/product/translation_in/goldstandard_eng_v1_utf8_names_casesensitive-generated.txt
'''

if __name__ == "__main__":
    if sys.argv[1]=="encode":
        encode_and_compress(sys.argv[2])
    elif sys.argv[1]=="ft":
        fine_tune(sys.argv[2])
    elif sys.argv[1]=="g":
        generate(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=="i":
        insert_into_data(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                         sys.argv[5], sys.argv[6], sys.argv[7])