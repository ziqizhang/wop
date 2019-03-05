# test changes

import sys
import os
import datetime

import gensim
import numpy
from numpy.random import seed

seed(1)

from classifier import classifier_dnn_multi_input as dnn_classifier
from classifier import dnn_util as util
from exp.wop import exp_wop_cml as exp_util
import pandas as pd



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


def strtobool(param):
    pass


if __name__ == "__main__":

    for setting_file in os.listdir(sys.argv[1]):
        properties = exp_util.load_properties(sys.argv[1]+'/'+setting_file)
        home_dir = sys.argv[2]
        # this is the file pointing to the CSV file containing the profiles to classify, and the profile texts from which we need to extract features
        csv_training_text_data = home_dir+properties['training_text_data']

        # this is the folder containing other gazetteer based features that are already pre-extracted
        csv_training_other_feaures = home_dir+properties['training_other_features']
        if len(csv_training_other_feaures)==0:
            csv_training_other_feaures=None

        # this is the folder to save output to
        outfolder = home_dir+properties["output_folder"]

        # if true, classes with instances less than n_fold will be removed
        remove_rare_classes = strtobool(sys.argv[3])
        remove_no_desc_instances = strtobool(sys.argv[4])

        print("\n"+str(datetime.datetime.now()))
        print("loading embedding models...")
        # this the Gensim compatible embedding file
        dnn_embedding_file =home_dir+properties["embedding_file"]  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
        gensimFormat = ".gensim" in dnn_embedding_file
        if gensimFormat:
            pretrained_embedding_models = gensim.models.KeyedVectors.load(dnn_embedding_file, mmap='r')
        else:
            pretrained_embedding_models = gensim.models.KeyedVectors. \
                load_word2vec_format(dnn_embedding_file, binary=True)

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

        model_descriptors = [
            #"input=2d han_2dinput",
            "input=2d bilstm=100-False|dense=?-softmax|glv",
            "input=2d cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=?-softmax|glv"]

        #input=3d han_full|glv,
        #input=2d lstm=100-False|dense=?-softmax|glv

        # "scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv",
        # "scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]

        ######## dnn #######
        print("loading dataset...")
        df = pd.read_csv(csv_training_text_data, header=0, delimiter=";", quoting=0, encoding="utf-8",
                         ).as_matrix()
        df.astype(str)
        if remove_no_desc_instances:
            print("you have chosen to remove instances whose description are empty")
            df=exp_util.remove_empty_desc_instances(df, 5)

        y = df[:, int(properties['class_column'])]

        target_classes = len(set(y))
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


        print('[STARTED] running settings with label='+properties['label'])

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

            input_column_sources = [x for x in properties['training_text_data_columns'].split("|")]
            # now create DNN branches based on the required input text column sources

            dnn_branches = []
            dnn_branch_input_shapes=[]
            dnn_branch_input_features=[]
            for string in input_column_sources:
                print("\tcreating model branch="+string)
                config = string.split(",")
                col_index=int(config[0])
                col_name=config[1]
                col_text_length=int(config[2])
                text_data = df[:, col_index]
                text_data = numpy.delete(text_data, remove_instance_indexes)
                data = ["" if type(x) is float else x for x in text_data]

                dnn_branch=dnn_classifier.create_dnn_branch(
                    pretrained_embedding_models,input_text_data=data,
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
            final_model=\
                dnn_classifier.merge_dnn_branch(dnn_branches, dnn_branch_input_shapes,
                                                target_classes)
            print("fitting model...")
            dnn_classifier.fit_dnn(inputs=dnn_branch_input_features,
                                   nfold=n_fold,
                                   y_train=y,
                                   final_model=final_model,
                                   outfolder=outfolder,
                                   task=properties['label'],
                                   model_descriptor=model_descriptor)
            print("Completed running all models on this setting file")
            print(datetime.datetime.now())

