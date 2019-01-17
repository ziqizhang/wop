'''
Reads a csv file as filter (/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper1/analysis/raw/top100_users_or_tweets.csv),
produces separate 'interaction.csv' files for chronic, acute, communicable, and non-communicable diseases
'''

import pandas as pd
import csv

def read_filters(csv_file):
    df = pd.read_csv(csv_file, header=0, delimiter=",", quoting=0).as_matrix()
    communicable=[]
    noncommunicable=[]
    chronic=[]
    acute=[]

    for row in df:
        tag = row[0]
        com = row[3]
        ch = row[4]
        if com=='c':
            communicable.append(tag)
        elif com=='uc':
            noncommunicable.append(tag)

        if ch=='ch':
            chronic.append(tag)
        elif ch=='ac':
            acute.append(tag)

    return communicable, noncommunicable, chronic, acute

def output_csvs(filter_tags, interaction_csv_file, out_file):
    df = pd.read_csv(interaction_csv_file, header=0, delimiter=",", quoting=0).as_matrix()

    with open(out_file, 'w') as f:
        cw = csv.writer(f, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cw.writerow(['tag','total','%_retweeted','avg_rt_freq','%_liked','avg_like_freq','as_replies','as_quotes'])

        for row in df:
            if row[0] in filter_tags:
                cw.writerow(row)

if __name__ == "__main__":
    csv_filter_file="/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper1/analysis/raw/top100_users_or_tweets.csv"
    csv_interaction_file="/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper1/analysis/raw/interaction_cleaned.csv"
    out_folder="/home/zz/Cloud/GDrive/ziqizhang/project/msm4phi/paper1/analysis/raw"

    communicable, noncommunicable, chronic, acute=read_filters(csv_filter_file)
    output_csvs(communicable, csv_interaction_file, out_folder+"/_communicable.csv")
    output_csvs(noncommunicable, csv_interaction_file, out_folder + "/_noncommunicable.csv")
    output_csvs(chronic, csv_interaction_file, out_folder + "/_chronic.csv")
    output_csvs(acute, csv_interaction_file, out_folder + "/_acute.csv")


