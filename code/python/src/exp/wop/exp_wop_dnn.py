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

seed(1)

from classifier import classifier_dnn_multi_input as dnn_classifier
from classifier import dnn_util as util
from exp.wop import exp_wop_cml as exp_util
from categories import cluster_categories as cc
import pandas as pd


def merge(input_columns: list, df: pd.DataFrame):
    to_merge = []
    for idx in input_columns:
        data = df[:, idx]
        data = ["" if type(x) is float else x for x in data]
        to_merge.append(data)

    texts = []
    for i in range(len(to_merge[0])):
        entry = ""
        for l in to_merge:
            entry += l[i] + ". "
        texts.append(entry.strip())
    return texts


def describe_task(properties, overwrite_params, setting_file):
    desc = 'setting_file='+os.path.splitext(os.path.basename(setting_file))[0]
    desc += '|embedding='
    desc += os.path.splitext(os.path.basename(
        load_setting('embedding_file', properties, overwrite_params)))[0]
    desc += '|training_text_data='
    desc += os.path.splitext(os.path.basename(
        load_setting('training_text_data', properties, overwrite_params)))[0]
    return desc

def run_single_setting(setting_file, home_dir, remove_rare_classes,
                       remove_no_desc_instances,
                       overwrite_params=None,
                       gensimFormat=None):
    properties = exp_util.load_properties(setting_file)

    csv_training_text_data = home_dir + load_setting('training_text_data', properties, overwrite_params)
    # this is the folder containing other numeric features that are already pre-extracted
    csv_training_other_feaures = home_dir + load_setting('training_other_features', properties, overwrite_params)

    # this is the folder to save output to
    outfolder = home_dir + load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file = home_dir + load_setting("embedding_file", properties,
                                                overwrite_params)  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    if gensimFormat is None:
        gensimFormat = ".gensim" in dnn_embedding_file
    if gensimFormat:
        pretrained_embedding_models = gensim.models.KeyedVectors.load(dnn_embedding_file, mmap='r')
    else:
        pretrained_embedding_models = gensim.models.KeyedVectors. \
            load_word2vec_format(dnn_embedding_file, binary=True)

    n_fold = int(load_setting("n_fold", properties, overwrite_params))

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

    model_descriptors = [
        "input=2d bilstm=100-False|dense=?-softmax|emb",
        "input=2d cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=?-softmax|emb",
        "input=2d han_2dinput"]
    # model_descriptors = [
    #     "input=2d han_2dinput"]

    # input=3d han_full|glv,
    # input=2d lstm=100-False|dense=?-softmax|glv

    # "scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv",
    # "scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]

    ######## dnn #######
    print("loading dataset...")
    df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     ).as_matrix()
    df.astype(str)
    if remove_no_desc_instances:
        print("you have chosen to remove instances whose description are empty")
        df = exp_util.remove_empty_desc_instances(df, 5)

    y = df[:, int(load_setting("class_column", properties, overwrite_params))]

    target_classes = len(set(y))
    print("\ttotal classes=" + str(target_classes))
    remove_instance_indexes = []
    if remove_rare_classes:
        print("you have chosen to remove classes whose instances are less than n_fold")
        instance_labels = list(y)
        class_dist = {x: instance_labels.count(x) for x in instance_labels}
        remove_labels = []
        for k, v in class_dist.items():
            if v < n_fold:
                remove_labels.append(k)
        remove_instance_indexes = []
        for i in range(len(y)):
            label = y[i]
            if label in remove_labels:
                remove_instance_indexes.append(i)
        y = numpy.delete(y, remove_instance_indexes)
        target_classes = len(set(y))

    print('[STARTED] running settings with label=' + load_setting("label", properties, overwrite_params))

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

        input_column_sources = \
            [x for x in load_setting("training_text_data_columns", properties, overwrite_params).split("|")]
        # now create DNN branches based on the required input text column sources

        dnn_branches = []
        dnn_branch_input_shapes = []
        dnn_branch_input_features = []
        for string in input_column_sources:
            print("\tcreating model branch=" + string)
            config = string.split(",")
            col_name = config[1]

            if col_name == "cat_cluster":  # input are numeric number not text so needs to be processed differently from text
                dnn_branch = dnn_classifier.create_dnn_branch_rawfeatures(
                    input_data_cols=config[0].split("-"),
                    dataframe_as_matrix=df
                )
            else:
                text_data = cc.create_text_input_data(config[0], df)

                col_text_length = int(config[2])

                text_data = numpy.delete(text_data, remove_instance_indexes)
                data = ["" if type(x) is float else str(x) for x in text_data]

                dnn_branch = dnn_classifier.create_dnn_branch_textinput(
                    pretrained_embedding_models, input_text_data=data,
                    input_text_sentence_length=col_text_length,
                    input_text_word_embedding_dim=util.DNN_EMBEDDING_DIM,
                    model_descriptor=model_descriptor,
                    embedding_trainable=False,
                    embedding_mask_zero=dnn_embedding_mask_zero
                )

            dnn_branches.append(dnn_branch[0])
            dnn_branch_input_shapes.append(dnn_branch[1])
            dnn_branch_input_features.append(dnn_branch[2])

        print("creating merged model (if multiple input branches)")
        final_model = \
            dnn_classifier.merge_dnn_branch(dnn_branches, dnn_branch_input_shapes,
                                            target_classes)
        print("fitting model...")
        dnn_classifier.fit_dnn(inputs=dnn_branch_input_features,
                               nfold=n_fold,
                               y_train=y,
                               final_model=final_model,
                               outfolder=outfolder,
                               task=describe_task(properties, overwrite_params,setting_file),
                               model_descriptor=model_descriptor)
        print("Completed running all models on this setting file")
        print(datetime.datetime.now())


# the program can overwrite parameters defined in setting files. for example, if you want to overwrite
# the embedding file, you can include this as an overwrite param in the command line, but specifying
# [embedding_file= ...] where 'embedding_file' must match the parameter name. Note that this will apply
# to ALL settings
def parse_overwrite_params(argv):
    params = {}
    for a in argv:
        if "=" in a:
            values = a.split("=")
            params[values[0]] = values[1]
    return params


def load_setting(param_name, properties: {}, overwrite_params: {} = None):
    if overwrite_params is not None and param_name in overwrite_params.keys():
        return overwrite_params[param_name]
    else:
        return properties[param_name]


if __name__ == "__main__":
    # argv-1: folder containing all settings to run, see 'input' folder
    # argv-2: working directory
    # argv3,4:set to False

    # the program can take additional parameters to overwrite existing ones defined in setting files.
    # for example, if you want to overwrite the embedding file, you can include this as an overwrite
    # param in the command line, but specifying [embedding_file= ...] where 'embedding_file'
    # must match the parameter name. Note that this will apply to ALL settings
    overwrite_params = parse_overwrite_params(sys.argv)

    for file in os.listdir(sys.argv[1]):
        gc.collect()

        print("now processing config file=" + file)
        setting_file = sys.argv[1] + '/' + file

        run_single_setting(setting_file, sys.argv[2], strtobool(sys.argv[3]),
                           strtobool(sys.argv[4]),
                           overwrite_params=overwrite_params)
