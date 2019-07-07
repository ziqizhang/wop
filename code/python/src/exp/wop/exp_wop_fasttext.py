#class used to run fasttext over wop data

# test changes
# WARNING: if using HAN, must use tensorflow, not theano!!

import sys
import os
import datetime
from distutils.util import strtobool
import gc
import gensim
import numpy
from numpy.random import seed
from exp import exp_util

seed(1)

from classifier import classifier_dnn_multi_input as dnn_classifier
from exp.wop import exp_wop_cml as exp_util
from categories import cluster_categories as cc
import pandas as pd


def run_single_setting(setting_file, home_dir, remove_rare_classes,
                       remove_no_desc_instances,
                       overwrite_params=None,
                       gensimFormat=None):
    properties = exp_util.load_properties(setting_file)

    csv_training_text_data = home_dir + exp_util.load_setting('training_text_data', properties, overwrite_params)
    # this is the folder containing other numeric features that are already pre-extracted
    csv_training_other_feaures = home_dir + exp_util.load_setting('training_other_features', properties, overwrite_params)

    # this is the folder to save output to
    outfolder = home_dir + exp_util.load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file = home_dir + exp_util.load_setting("embedding_file", properties,
                                                overwrite_params)  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    if gensimFormat is None:
        gensimFormat = ".gensim" in dnn_embedding_file
    if gensimFormat:
        pretrained_embedding_models = gensim.models.KeyedVectors.load(dnn_embedding_file, mmap='r')
    else:
        pretrained_embedding_models = gensim.models.KeyedVectors. \
            load_word2vec_format(dnn_embedding_file, binary=True)

    n_fold = int(exp_util.load_setting("n_fold", properties, overwrite_params))

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

    ######## dnn #######
    print("loading dataset...")
    df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     ).as_matrix()
    df.astype(str)
    if remove_no_desc_instances:
        print("you have chosen to remove instances whose description are empty")
        df = exp_util.remove_empty_desc_instances(df, 5)

    y_train = df[:, int(exp_util.load_setting("class_column", properties, overwrite_params))]

    target_classes = len(set(y_train))
    print("\ttotal classes=" + str(target_classes))
    remove_instance_indexes = []
    if remove_rare_classes:
        print("you have chosen to remove classes whose instances are less than n_fold")
        instance_labels = list(y_train)
        class_dist = {x: instance_labels.count(x) for x in instance_labels}
        remove_labels = []
        for k, v in class_dist.items():
            if v < n_fold:
                remove_labels.append(k)
        remove_instance_indexes = []
        for i in range(len(y_train)):
            label = y_train[i]
            if label in remove_labels:
                remove_instance_indexes.append(i)
        y_train = numpy.delete(y_train, remove_instance_indexes)
        target_classes = len(set(y_train))

    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    input_column_sources = exp_util.load_setting("training_text_data_columns", properties, overwrite_params)
    text_data = cc.create_text_input_data(input_column_sources.split(","), df)
    # now create DNN branches based on the required input text column sources

    text_data = numpy.delete(text_data, remove_instance_indexes)
    X_train = ["" if type(x) is float else str(x) for x in text_data]

    print("fitting model...")
    dnn_classifier.fit_fasttext(X_train=X_train,y_train=y_train,
                               nfold=n_fold,
                               outfolder=outfolder,
                               task=exp_util.describe_task(properties, overwrite_params,setting_file))
    print("Completed running all models on this setting file")
    print(datetime.datetime.now())


if __name__ == "__main__":
    # argv-1: folder containing all settings to run, see 'input' folder
    # argv-2: working directory
    # argv3,4:set to False

    # the program can take additional parameters to overwrite existing ones defined in setting files.
    # for example, if you want to overwrite the embedding file, you can include this as an overwrite
    # param in the command line, but specifying [embedding_file= ...] where 'embedding_file'
    # must match the parameter name. Note that this will apply to ALL settings
    overwrite_params = exp_util.parse_overwrite_params(sys.argv)

    for file in os.listdir(sys.argv[1]):
        gc.collect()

        print("now processing config file=" + file)
        setting_file = sys.argv[1] + '/' + file

        run_single_setting(setting_file, sys.argv[2], strtobool(sys.argv[3]),
                           strtobool(sys.argv[4]),
                           overwrite_params=overwrite_params,gensimFormat=True)
