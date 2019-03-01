# test changes

import sys
import os
import datetime
from numpy.random import seed

seed(1)

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd


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

def merge(input_columns:list, df: pd.DataFrame):
    to_merge = []
    for idx in input_columns:
        data = df[:, idx]
        data = ["" if type(x) is float else x for x in data]
        to_merge.append(data)

    texts=[]
    for i in range(len(to_merge[0])):
        entry=""
        for l in to_merge:
            entry+=l[i]+". "
        texts.append(entry.strip())
    return texts


if __name__ == "__main__":

    for setting_file in os.listdir(sys.argv[1]):
        properties = load_properties(sys.argv[1]+'/'+setting_file)
        # this is the file pointing to the CSV file containing the profiles to classify, and the profile texts from which we need to extract features
        csv_training_text_data = properties['training_text_data']
        # ProductCategorisation: [GoogleDrive]/wop/datasets/ProductCategorisation/goldstandard_eng_v1.csv
        # Twitter: [your path]/wop/data/ml/training_text_features.csv

        # this is the folder containing other gazetteer based features that are already pre-extracted
        csv_training_other_feaures = properties['training_other_features']
        if len(csv_training_other_feaures)==0:
            csv_training_other_feaures=None
        # ProductCategorisation: N/A
        # Twitter: [your path]/wop/data/ml/training_other_features/gazetteer/dict1_match_to_profile.csv

        # this is the folder to save output to
        outfolder = properties["output_folder"]

        # this the Gensim compatible embedding file
        dnn_embedding_file = properties["embedding_file"]  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"

        n_fold = int(properties["n_fold"])

        # in order to test different DNN architectures, I implemented a parser that analyses a string following
        # specific syntax, creates different architectures. This one here takes word embedding, pass it to 3
        # cnn layer then concatenate the output by max pooling finally into a softmax
        #
        # So you can add mulitple descriptors in to a list, and the program will go through each model structure, apply them
        # to the same dataset for experiments
        #
        # the descriptor is passed as a param to 'Classifer', which parses the string to create a model
        # see 'classifier_learn.py - learn_dnn method for details
        # todo: when HAN used, metafeature must NOT be set
        model_descriptors = ["input=3d han_full|glv",
                             "input=2d cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=?-softmax|glv"]
        # ["han_full"]
        # "scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv",
        # "scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]

        ######## dnn #######
        print(datetime.datetime.now())

        print('Running settings with label='+properties['label'])

        for model_descriptor in model_descriptors:
            print("\tML model=" + model_descriptor)

            input_shape = model_descriptor.split(" ")[0]
            model_descriptor = model_descriptor.split(" ")[1]

            if input_shape.endswith("2d"):
                input_as_2D = True
            else:
                input_as_2D = False

            if "han" in model_descriptor or "lstm" in model_descriptor:
                dnn_embedding_mask_zero = True
            else:
                dnn_embedding_mask_zero = False

            # For ProductCategorisation dataset
            # In this experiment, I used both the title (index 4) and the description (index 5) of each item,
            # because title is generally very short and the description is not available for all items.
            input_columns = [int(x) for x in properties['training_text_data_columns'].split(",")]
            X, y = fc.create_features_text_multiple_fields(csv_training_text_data,
                                                           text_cols=input_columns,
                                                           label_col=int(properties['class_column']),
                                                           data_delimiter=";",
                                                           text_encoding="utf-8", text_header=0)
            # If one field needs to be used instead, please replace the above line with the following:
            # X, y = fc.create_features_text(csv_training_text_data, text_col=4, label_col=10, data_delimiter=";",
            #                                text_encoding="ANSI", text_header=0)
            df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                             ).as_matrix()
            df.astype(str)

            # Merge the specified text data fields in the input training_text_csv file to create the concatenated
            # text fields
            texts=merge(input_columns, df)

            cls = cm.Classifer(properties['label'], "_dnn_", None, y, outfolder,
                               categorical_targets=int(properties['classes']),
                               algorithms=["dnn"], nfold=n_fold,
                               text_data=texts, dnn_embedding_file=dnn_embedding_file,
                               dnn_descriptor=model_descriptor, dnn_input_as_2D=input_as_2D,
                               dnn_embedding_trainable=False,
                               dnn_embedding_mask_zero=dnn_embedding_mask_zero)
            cls.run()

            print(datetime.datetime.now())

            # X would be the 'metafeature' to pass to the dnn model. Note it MUST NOT contain text and should be
            # ready-to-use features.
            # WARNING: using metafeature with 3D input shape is currently NOT supported. The code will ignore the metafeature
            # see classifier_learn.py line 226
            # X, y = fc.create_features_gazetteer(csv_training_text_data, csv_training_other_feaures)
            # df = pd.read_csv(csv_training_text_data, header=0, delimiter=",", quoting=0).as_matrix()
            # df.astype(str)
            # profiles = df[:, 22]
            # profiles = ["" if type(x) is float else x for x in profiles]
            # cls = cm.Classifer("stakeholdercls", "_dnn_text+other_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
            #                    text_data=profiles, dnn_embedding_file=dnn_embedding_file,
            #                    dnn_descriptor=model_descriptor,
            #                    dnn_text_data_extra_for_embedding_vcab=tweets_exta)
            # cls.run()

            # #
