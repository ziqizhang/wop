# Given the original user_features.csv file that is a simple dump of indexed user features,
# this code processes it further to extract different kinds of features based on dictionaries
import csv
import re
import os

import datetime
import pandas as pd
from feature import dictionary_extractor as de
from feature import nlp
from feature import dictionary_extractor_dhashtag as dedh

# input: a dictionary containing different dictionaries to be used
def match_extracted_dictionary(dictionaries: dict, csv_input_feature_file, col_id, outfile,
                               *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    dict_labels = list(dictionaries.keys())
    for k in dict_labels:
        output_header.append(k + "_scoresum")
        output_header.append(k + "_matchsum")
        output_header.append(k + "_matchmax")
        output_header.append(k + "_matchbool")
    output_matrix.append(output_header)

    count=0
    for row in df:
        #print(count)
        count+=1
        row_data = [row[col_id]]
        target_text = ""

        skip=False
        for tt_col in col_target_texts:
            text = row[tt_col]
            if type(text) is float:
                skip=True
                break

            target_text += text + " "

        if skip:
            for k in dict_labels:
                row_data.append("0")
                row_data.append("0")
                row_data.append("0")
                row_data.append("0")
            output_matrix.append(row_data)
            continue
        target_text = target_text.strip()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        for k in dict_labels:
            dictionary = dictionaries[k]
            scoresum, matchsum, matchmax, matchbool = \
                find_word_matches(dictionary, target_text, de.text_normalization_option)
            row_data.append(scoresum)
            row_data.append(matchsum)
            row_data.append(matchmax)
            row_data.append(matchbool)
        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


def match_extracted_healthconditions(dictionary: dict, csv_input_feature_file, col_id, outfile,
                                     *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()

    output_matrix = []

    output_header = ["user_id"]
    output_header.append("has_hc")
    output_header.append("count_hc")
    output_matrix.append(output_header)

    for row in df:
        row_data = [row[col_id]]
        target_text = ""

        skip=False
        for tt_col in col_target_texts:
            text = row[tt_col]
            if type(text) is float:
                skip=True
                break
            target_text += row[tt_col] + " "

        if skip:
            row_data.append("0")
            row_data.append("0")
            output_matrix.append(row_data)
            continue

        target_text = target_text.strip().lower()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        count_hc = find_hc_matches(dictionary, target_text)

        has_hc = 0
        if count_hc > 0:
            has_hc = 1
        row_data.append(has_hc)
        row_data.append(count_hc)
        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


# person name, profession, title; against name, profile etc.
def match_generic_gazetteer(dictionaries: dict, csv_input_feature_file, col_id, outfile,
                            *col_target_texts):
    df = pd.read_csv(csv_input_feature_file, header=0, delimiter=",", quoting=0).as_matrix()
    profession_regex = r'ist\b'

    output_matrix = []

    output_header = ["user_id"]
    dict_labels = list(dictionaries.keys())
    for k in dict_labels:
        output_header.append(k + "_hasmatch")
    output_matrix.append(output_header)


    for row in df:
        row_data = [row[col_id]]
        target_text = ""

        skip=False
        for tt_col in col_target_texts:
            text = row[tt_col]
            if type(text) is float:
                skip=True
                break;
            target_text += row[tt_col] + " "

        if skip:
            for k in dict_labels:
                row_data.append("0")
            output_matrix.append(row_data)
            continue
        target_text = target_text.strip().lower()

        if len(target_text) < 2:
            output_matrix.append(row_data)
            continue

        toks = set(target_text.split(" "))
        for k in dict_labels:
            dictionary = dictionaries[k]
            if len(toks.intersection(dictionary)) > 0:
                row_data.append("1")
            else:
                if 'profession' in k and len(re.findall(profession_regex, target_text))>0:
                    row_data.append("1")
                else:
                    row_data.append("0")



        output_matrix.append(row_data)

    with open(outfile, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                               quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in output_matrix:
            csvwriter.writerow(row)


def find_hc_matches(dictionary: dict, target_text):
    hashtag_regex = '#[\w\-]+'
    matches = re.findall(hashtag_regex, target_text)
    found=False
    for m in matches:
        target_text += " " + m[1:]
        found=True
    target_text = target_text.strip()

    hc = set()

    if(found):
        toks = set(target_text.split(" "))

        inter = toks.intersection(set(dictionary.keys()))
        for t in inter:
            hc.update(dictionary[t])

    return len(hc)


def find_word_matches(dictionary, target_text, text_normalization_option):
    target_text = nlp.normalize_tweet(target_text)
    norm_toks = set(nlp.tokenize(target_text, text_normalization_option))

    scoresum = 0
    matchsum = 0
    matchmax = 0
    matchbool = 0
    for w, score in dictionary.items():
        score=float(score)
        if w in norm_toks:
            matchbool = 1
            matchsum += 1
            scoresum += score
            if matchmax < score:
                matchmax = score

    return scoresum, matchsum, matchmax, matchbool

# Profile: pattern, e.g., 'VERB X X HealthCondition'
def flatten_dictionary(postype_dictionaries):
    out_dict={}
    for postype, dictionaries in postype_dictionaries.items():
        for label, dicts in dictionaries.items():
            if label=="Other":
                continue
            out_dict[postype+"_"+label]=dicts
    return out_dict

def load_generic_dictionary(txtfile):
    with open(txtfile) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().lower() for x in content]
    return content

if __name__=="__main__":
    #folder containing the dictionaries
    dictionary_folder="/home/zz/Work/msm4phi/resources/dictionary"
    #original feature csv file containing at least the text fields to be matched, and user id

    #dict1-dictionary created based on lemmatisation; dict2-based on stemming
    #therefore also change the value'text_normalization_option = 0' in dictionary_extractor to use corresponding normalisation on text
    dict_lemstem_option="dict1"
    # output folder to save dictionary features
    # outfolder = "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/features/full/dictionary_feature_1"
    # csv_feature_input = "/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper2/data/training_data/basic_features.csv"
    outfolder = "/home/zz/Work/msm4phi_data/paper2/all_user_empty_filled_autodictext_features"
    csv_feature_input = "/home/zz/Work/msm4phi_data/paper2/all_user_empty_filled_features"


    # column id of the target text field
    target_text_cols = 16  # 22=profile text; 15=name field
    target_text_name_suffix="_profile"
    col_id=0
    #how many entries from each dictionary should be selected for matching(top n). Changing this param will generate
    #different features, so perhaps influencing classification results
    topN_of_dict=100

    for file in os.listdir(csv_feature_input):

        prefix=file[0:file.index(".csv")]+"_"

        print(file)
        print(datetime.datetime.now())
        file=csv_feature_input+"/"+file
        #load auto extracted dictionaries, match to 'profile'
        postype_dictionaries = \
            de.load_extracted_dictionary(dictionary_folder+"/auto_created/profile/"+dict_lemstem_option+"/frequency_pass2",
                                         topN_of_dict, "verb", "noun")
        extracted_dictionaries = flatten_dictionary(postype_dictionaries)
        match_extracted_dictionary(extracted_dictionaries, file,
                                   col_id, outfolder +"/"+prefix+"feature_autocreated_dict_match" + target_text_name_suffix +".csv",
                                   target_text_cols)

        #load hashtag dictionaries
        hashtag_dictionary = dedh.load_disease_hashtag_dictionary(
            dictionary_folder+"/hashtag_dict/dictionary_hashtag_disease.csv"
        )
        match_extracted_healthconditions(hashtag_dictionary, file, col_id,
                                         outfolder +"/"+prefix+"feature_disease_hashtag_match" + target_text_name_suffix +".csv",
                                         target_text_cols)

        disease_word_dictionary=dedh.load_disease_hashtag_dictionary(
            dictionary_folder+"/hashtag_dict/dictionary_word_disease.csv"
        )
        match_extracted_healthconditions(disease_word_dictionary, file, col_id,
                                         outfolder + "/"+prefix+"feature_disease_word_match" + target_text_name_suffix +".csv",
                                         target_text_cols)


        #load other generic dictionaries
        #person name
        #person_name_dict=load_generic_dictionary(dictionary_folder+"/name/person_names.txt")
        #person title
        person_title_dict = load_generic_dictionary(dictionary_folder+"/manually_created/generic/person_titles.txt")
        #profession
        person_profession_dict = load_generic_dictionary(dictionary_folder+"/manually_created/generic/person_professions.txt")
        generic_dict={}
        #person name should only be used to match against the 'name' fields
        #generic_dict["person_name"]=person_name_dict
        generic_dict["person_title"]=person_title_dict
        generic_dict["person_profession"]=person_profession_dict
        # match_generic_gazetteer(generic_dict,csv_input_feature_file,
        #                          col_id, outfolder+"/feature_generic_dict_match"+target_text_name_suffix+".csv",
        #                          target_text_cols)

