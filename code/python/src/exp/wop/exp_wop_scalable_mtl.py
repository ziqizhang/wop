# use this class to run experiments over wop datasets with nfold validation
import sys
import datetime

import numpy
import tensorflow as tf
from keras import initializers

from lanmodel import embedding_util

from classifier import classifier_dnn_scalable as dnn_classifier
from classifier import mtl_util as mtl_classifier
from classifier import dnn_util as util
from exp.wop import exp_wop_cml as exp_util
import pandas as pd
from exp import exp_util
from classifier import classifier_learn
import random

random.seed(classifier_learn.RANDOM_STATE)
numpy.random.seed(classifier_learn.RANDOM_STATE)
tf.set_random_seed(classifier_learn.RANDOM_STATE)

my_init = initializers.glorot_uniform(seed=classifier_learn.RANDOM_STATE)


# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                               inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)


def load_word_weights(word_weights_file):
    weights = pd.read_csv(word_weights_file, delimiter=",", quoting=0, encoding="utf-8",
                          ).as_matrix()
    words = []
    for r in weights:
        words.append(r[0])
    return words


def run_mtl_setting(setting_file, home_dir,
                    overwrite_params=None,
                    embedding_format=None):
    properties = exp_util.load_properties(setting_file)

    word_weights_file = exp_util.load_setting('word_weights_file', properties, overwrite_params)
    if word_weights_file == None:
        word_weights = None
    else:
        print("using word weights to revise embedding vectors...")
        word_weights = load_word_weights(word_weights_file)

    csv_training_text_data = home_dir + exp_util.load_setting('training_text_data', properties, overwrite_params)

    # this is the folder to save output to
    outfolder = home_dir + exp_util.load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file = home_dir + exp_util.load_setting("embedding_file", properties,
                                                          overwrite_params)  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    # print("embedding file is========="+dnn_embedding_file)
    emb_model = embedding_util.load_emb_model(embedding_format, dnn_embedding_file)

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
        # "input=2d cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=?-softmax|emb",
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

    #stats about main task
    maintask_class_col = int(exp_util.load_setting("class_column", properties, overwrite_params))
    main_y = df[:, maintask_class_col]
    target_classes = len(set(main_y))
    print("\ttotal classes=" + str(target_classes))

    #stats about auxiliary tasks
    auxtask_class_col=exp_util.load_setting("class_auxiliary", properties, overwrite_params)
    if auxtask_class_col==None:
        print("Not MTL, quit.")
        exit(1)
    auxtask_class_cols=[]
    aux_classes=[]
    for i in auxtask_class_col.split(","):
        i=int(i)
        aux_y = df[:, i]
        aux_cls = len(set(aux_y))
        print("\t\t auxiliary task with classes=" + str(aux_classes))
        auxtask_class_cols.append(i)
        aux_classes.append(aux_cls)


    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    for model_descriptor in model_descriptors:
        print("\tML shared model=" + model_descriptor)

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

        print("creating MTL model (if multiple input branches)")
        final_model = \
            mtl_classifier.create_mtl_layers(dnn_branches, dnn_branch_input_shapes,
                                            target_classes, aux_classes)
        print("fitting model...")

        mtl_classifier.fit_dnn_mtl(df=df,
                                   nfold=n_fold,
                                   main_class_col=maintask_class_col,
                                   aux_class_cols=auxtask_class_cols,
                                   final_model=final_model,
                                   outfolder=outfolder,
                                   task=exp_util.describe_task(properties, overwrite_params, setting_file),
                                   model_descriptor=model_descriptor, text_norm_option=1,
                                   text_input_info=input_text_info,
                                   embedding_model=emb_model,
                                   embedding_model_format=embedding_format,
                                   word_weights=word_weights)
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

    setting_file = sys.argv[1]

    if sys.argv[4] == 'dnn':
        run_mtl_setting(setting_file, sys.argv[2],
                        overwrite_params=overwrite_params, embedding_format=sys.argv[3])
    else:
        print("Not supported")



