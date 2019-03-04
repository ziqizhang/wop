'''
experiment runner class for WOP using classic ML algorithms
'''

import sys
import datetime
import os
import pandas as pd
import numpy

from exp import feature_creator as fc
from classifier import classifier_main as cm
from feature import text_feature_extractor as tfe


def load_properties(filepath, sep='=', comment_char='#'):
    """
    Read the file passed as parameter as a properties file.
    """
    props = {}
    with open(filepath, "rt") as f:
        for line in f:
            l = line.strip()
            if l and not l.startswith(comment_char):
                key_value = l.split(sep)
                key = key_value[0].strip()
                value = sep.join(key_value[1:]).strip().strip('"')
                props[key] = value
    return props

if __name__ == "__main__":
    for setting_file in os.listdir(sys.argv[1]):
        properties = load_properties(sys.argv[1]+'/'+setting_file)
        #to use this class, your home directory must containt he follows:
        #- wop: containing this project folder
        #- data: containing word embedding models
        #- data/wop_data: containing gold standard datae

        home_dir = sys.argv[2]
        # this is the file pointing to the CSV file containing the profiles to classify, and the profile texts from which we need to extract features
        csv_training_text_data = home_dir+properties['training_text_data']

        # this is the folder containing other gazetteer based features that are already pre-extracted
        csv_training_other_feaures = home_dir+properties['training_other_features']
        if len(csv_training_other_feaures)==0:
            csv_training_other_feaures=None

        # this is the folder to save output to
        outfolder = home_dir+properties["output_folder"]
        n_fold = int(properties["n_fold"])

        print("loading dataset...")
        df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                         ).as_matrix()
        df.astype(str)
        y = df[:, int(properties['class_column'])]

        print('[STARTED] running settings with label='+properties['label'])

        input_column_sources = [x for x in properties['training_text_data_columns'].split("|")]
        features_from_separate_fields=[]
        for string in input_column_sources:
            print("\textracting features from: " + string)
            config = string.split(",")
            col_index = int(config[0])
            col_name = config[1]
            text_data = df[:, col_index]
            data = ["" if type(x) is float else x for x in text_data]
            X_ngram, vocab = tfe.get_ngram_tfidf(data)
            features_from_separate_fields.append(X_ngram)
        X_all = numpy.concatenate(features_from_separate_fields, axis=1)

        # print("\tfeature extraction completed.")
        # print(datetime.datetime.now())
        # print("\nRunning nb")
        # cls = cm.Classifer(properties['label'], "nb", X_all, y, outfolder,
        #                    categorical_targets=int(properties["classes"]),
        #                    nfold=n_fold, algorithms=["nb"])
        # cls.run()

        print(datetime.datetime.now())
        print("\nRunning pca-svm_l")
        cls = cm.Classifer(properties['label'], "svm_l", X_all, y, outfolder,
                           categorical_targets=int(properties["classes"]),
                           nfold=n_fold, algorithms=["svm_l"])
        cls.run()

        # print(datetime.datetime.now())
        # print("\nRunning pca-knn")
        # cls = cm.Classifer(properties['label'], "knn", X_all, y, outfolder,
        #                    categorical_targets=int(properties["classes"]),
        #                    nfold=n_fold, algorithms=["knn"])
        # cls.run()

        # print(datetime.datetime.now())
        # print("\nRunning pca-lr")
        # cls = cm.Classifer(properties['label'], "pca-lr", X_all, y, outfolder,
        #                    categorical_targets=int(properties["classes"]),
        #                    nfold=n_fold, algorithms=["pca-lr"])
        # cls.run()
        #
        # print(datetime.datetime.now())
        # print("\nRunning pca-rf")
        # cls = cm.Classifer(properties['label'], "pca-rf", X_all, y, outfolder,
        #                    categorical_targets=int(properties["classes"]),
        #                    nfold=n_fold, algorithms=["pca-rf"])
        # cls.run()


