# use this class to run experiments over wop datasets with nfold validation

import sys
import datetime
from distutils.util import strtobool
import gensim
from fasttext import load_model
from numpy.random import seed

seed(1)

from classifier import classifier_dnn_scalable as dnn_classifier
from classifier import dnn_util as util
from exp.wop import exp_wop_cml as exp_util
import pandas as pd
from exp import exp_util


def run_dnn_setting(setting_file, home_dir,
                    overwrite_params=None,
                    embedding_format=None):
    properties = exp_util.load_properties(setting_file)

    csv_training_text_data = home_dir + exp_util.load_setting('training_text_data', properties, overwrite_params)

    # this is the folder to save output to
    outfolder = home_dir + exp_util.load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file = home_dir + exp_util.load_setting("embedding_file", properties,
                                                          overwrite_params)  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    #print("embedding file is========="+dnn_embedding_file)
    if embedding_format == 'gensim':
        print("\tgensim format")
        emb_model = gensim.models.KeyedVectors.load(dnn_embedding_file, mmap='r')
    elif embedding_format == 'fasttext':
        print("\tfasttext format")
        emb_model = load_model(dnn_embedding_file)
    else:
        print("\tword2vec format")
        emb_model = gensim.models.KeyedVectors. \
            load_word2vec_format(dnn_embedding_file, binary=strtobool(embedding_format))

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
                     )
    df = df.fillna('')
    df = df.as_matrix()
    class_col = int(exp_util.load_setting("class_column", properties, overwrite_params))
    y = df[:, class_col]

    target_classes = len(set(y))
    print("\ttotal classes=" + str(target_classes))

    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    for model_descriptor in model_descriptors:
        print("\tML model=" + model_descriptor)

        model_descriptor = model_descriptor.split(" ")[1]

        dnn_branches = []
        dnn_branch_input_shapes = []
        input_text_info = {}
        count = 0
        for x in exp_util.load_setting("training_text_data_columns", properties, overwrite_params).split("|"):
            config = x.split(",")
            map = {}
            map["text_col"] = config[0]
            map["text_length"] = int(config[2])
            map["text_dim"] = util.DNN_EMBEDDING_DIM
            input_text_info[count] = map
            dnn_branch = dnn_classifier.create_dnn_branch(map["text_length"],
                                                          util.DNN_EMBEDDING_DIM,
                                                          model_descriptor=model_descriptor
                                                          )
            dnn_branches.append(dnn_branch[0])
            dnn_branch_input_shapes.append(dnn_branch[1])
            count += 1
        # now create DNN branches based on the required input text column sources

        print("creating merged model (if multiple input branches)")
        final_model = \
            dnn_classifier.merge_dnn_branch(dnn_branches, dnn_branch_input_shapes,
                                            target_classes)
        print("fitting model...")

        dnn_classifier.fit_dnn(df=df,
                               nfold=n_fold,
                               class_col=class_col,
                               final_model=final_model,
                               outfolder=outfolder,
                               task=exp_util.describe_task(properties, overwrite_params, setting_file),
                               model_descriptor=model_descriptor, text_norm_option=1,
                               text_input_info=input_text_info,
                               embedding_model=emb_model,
                               embedding_model_format=embedding_format)
    print("Completed running all models on this setting file")
    print(datetime.datetime.now())


def run_fasttext_setting(setting_file, home_dir,
                         overwrite_params=None):
    properties = exp_util.load_properties(setting_file)

    csv_training_text_data = home_dir + exp_util.load_setting('training_text_data', properties, overwrite_params)

    # this is the folder to save output to
    outfolder = home_dir + exp_util.load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file = home_dir + exp_util.load_setting("embedding_file", properties,
                                                          overwrite_params)  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    if dnn_embedding_file.endswith('none'):
        dnn_embedding_file=None

    n_fold = int(exp_util.load_setting("n_fold", properties, overwrite_params))

    ######## dnn #######
    print("loading dataset...")
    df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                     )
    df = df.fillna('')
    df = df.as_matrix()
    class_col = int(exp_util.load_setting("class_column", properties, overwrite_params))
    y = df[:, class_col]

    target_classes = len(set(y))
    print("\ttotal classes=" + str(target_classes))

    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    print("fitting model...")

    input_text_info = {}
    count = 0
    for x in exp_util.load_setting("training_text_data_columns", properties, overwrite_params).split("|"):
        config = x.split(",")
        map = {}
        map["text_col"] = config[0]
        map["text_length"] = int(config[2])
        map["text_dim"] = util.DNN_EMBEDDING_DIM
        input_text_info[count] = map

    dnn_classifier.fit_fasttext(df=df,
                                nfold=n_fold,
                                class_col=class_col,
                                outfolder=outfolder,
                                task=exp_util.describe_task(properties, overwrite_params, setting_file),
                                text_norm_option=1,
                                text_input_info=input_text_info,
                                embedding_file=dnn_embedding_file)
    print("Completed running on this setting file")
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

    setting_file = sys.argv[1]

    if sys.argv[4] == 'fasttext':
        run_fasttext_setting(setting_file, sys.argv[2], overwrite_params=overwrite_params)
    else:
        run_dnn_setting(setting_file, sys.argv[2],
                        overwrite_params=overwrite_params, embedding_format=sys.argv[3])


