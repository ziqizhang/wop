'''
This file reads in the folder containing swc annotations, processes them and convert them into json format

'''
import pandas as pd
import json
import os

#create a dictionary that maps lvl3 label to lvl1 and lvl2
def read_original_taxonomy(original_csv):
    lvl1col=10
    lvl2col=11
    lvl3col=12
    df = pd.read_csv(original_csv, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     )
    df = df.fillna('')
    df = df.as_matrix()

    lvl3_to_lvl1 = dict()
    lvl3_to_lvl2=dict()
    for row in df:
        lvl1=row[lvl1col]
        lvl2=row[lvl2col]
        lvl3=row[lvl3col]
        lvl3_to_lvl1[lvl3]=lvl1
        lvl3_to_lvl2[lvl3]=lvl2
    return lvl3_to_lvl1, lvl3_to_lvl2

#convert the entire folders of annotation files
def convert(input_annotation_folder, input_original_csv, output_json_file):
    lvl3_to_lvl1, lvl3_to_lvl2=read_original_taxonomy(input_original_csv)
    writer = open(output_json_file,'w')

    selected_products=dict()
    selected_p_from=dict()
    freq=dict()

    alltotal=0
    for f in os.listdir(input_annotation_folder):
        f=input_annotation_folder+"/"+f
        if os.path.isfile(f):
            continue

        separator=';'
        if 'comma' in f:
            separator=','

        for af in os.listdir(f):
            if not af.endswith('.csv'):
                continue
            af=f+"/"+af
            print(af)
            try:
                total_from_file=process(af, separator,
                                    lvl3_to_lvl1,lvl3_to_lvl2,selected_products,selected_p_from,freq,writer)
            except UnicodeDecodeError:
                print("CANNOT OPEN FILE, pass")
                continue

            alltotal+=total_from_file

            print("\ttotal={}".format(total_from_file))


    writer.close()
    print("\n Overall={}\nSelectedProducts={}".format(alltotal, len(selected_products)))
    print(freq)


#process an individual annotation file
def process(annotation_file, separator, lvl3_to_lvl1:dict, lvl3_to_lvl2:dict, selected_pros:dict, selected_from:dict,
            freq:dict,
            writer):
    df = pd.read_csv(annotation_file, header=None, delimiter=separator, quoting=0, encoding="utf-8",
                     )
    df = df.fillna('')
    df = df.as_matrix()

    namecol=4
    lvl3col=5
    catcol=6
    urlcol=7
    desccol=8

    count=0
    for row in df:
        if row[0]=='Original_GS_name':
            continue
        name=row[namecol].strip()
        url = row[urlcol].strip()
        desc = row[desccol].strip()
        cat = row[catcol].strip()
        lvl3=row[lvl3col].strip()

        if len(lvl3)==0:
            continue

        if '_' not in lvl3:
            print("\t\tERROR: {} | \t\t {}".format(lvl3, name))
            continue

        if name in selected_pros.keys():
            _from=selected_from[name]
            annotation=selected_pros[name]
            if annotation==lvl3:
                continue
            else:
                print("\t\tINconsistent label for: {} | \t\t A={}, B={} from {}".format(name,lvl3, annotation, _from))
                continue

        lvl2=lvl3_to_lvl2[lvl3]
        lvl1=lvl3_to_lvl1[lvl3]

        if lvl1 is None or lvl2 is None:
            print("\t\t SEVERE error: lvl3 not found in mappings: {}".format(lvl3))
            continue

        count+=1

        if lvl3 in freq.keys():
            freq[lvl3]+=1
        else:
            freq[lvl3]=1


        selected_pros[name]=lvl3
        selected_from[name]=annotation_file
        js = {
            "ID": count,
            "Name": name,
            "Description": desc,
            "CategoryText":cat,
            "URL":url,
            "lvl1":lvl1,
            "lvl2":lvl2,
            "lvl3":lvl3
        }

        # convert into JSON:
        line = json.dumps(js)
        writer.write(line+"\n")

    return count


if __name__ == "__main__":
    original_csv = "/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv"
    annotation_folder = "/home/zz/Work/data/wop/swc/trial_annotation/Nov2017_omitNorms=false"
    out_file = "/home/zz/Work/data/wop/swc/swc_dataset/extended.json"
    convert(annotation_folder, original_csv, out_file)