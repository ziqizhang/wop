import pandas as pd
import sys
import csv

def extract_column(inCSV:str, separator:str, name_col:int, out_list_file):
    data = pd.read_csv(inCSV, header=0, delimiter=separator, quoting=0, encoding="utf-8",
                        ).fillna('').as_matrix()
    with open(out_list_file, 'w', newline='') as f:
        for row in data:
            name = row[name_col].lower()
            f.write(name+"\n")

def insert_into_data(inCSV:str, separator:str, add_to_col:int, out_translation, outCSV:str):
    data = pd.read_csv(inCSV, header=0, delimiter=separator, quoting=0, encoding="utf-8",
                       ).fillna('')
    headers= list(data.columns.values)
    if len(headers)<=add_to_col:
        headers.append("translated_label")

    data=data.as_matrix()
    with open(out_translation) as f:
        lineList = f.readlines()

    with open(outCSV, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=separator, quotechar='"')
        writer.writerow(headers)
        for i in range(len(data)):
            row=list(data[i])
            translation=lineList[i].strip()

            translation=translation.replace("_"," ")

            #only keep unique words
            words=list(translation.split(" "))
            translation=" ".join(words)
            #

            if len(row)>add_to_col:
                row[add_to_col]=translation
            else:
                row.append(translation)
            writer.writerow(row)

if __name__ == "__main__":
    if sys.argv[1]=="e":
        '''
        e
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv
;
4
/home/zz/Work/data/wop_data/mt/product/translation_in/goldstandard_eng_v1_utf8_names.txt


e
/home/zz/Work/data/Rakuten/rdc-catalog-train_fasttext.tsv
\t
1
/home/zz/Work/data/mt/product/translation_in/rdc-catalog-train_fasttext_name.txt

e
/home/zz/Work/data/Rakuten/rdc-catalog-gold_fasttext.tsv
\t
1
/home/zz/Work/data/mt/product/translation_in/rdc-catalog-gold_fasttext_name.txt

        '''
        extract_column(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        '''
        i
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv
;
13
/home/zz/Work/data/wop_data/mt/product/translation_out/goldstandard_eng_v1_utf8_names.txt
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_tl.csv

        i
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv
;
13
/home/zz/Work/data/wop_data/mt/product/translation_out/goldstandard_eng_v1_utf8_names_tll.txt
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_tll.csv
        '''
        insert_into_data(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                         sys.argv[5], sys.argv[6])