'''
Use this file to run experiments of cnn/lstm/han and svm/nb/knn etc on datasets split into train/val/test
'''
# use this class to run experiments over wop datasets with nfold validation
import sys
import datetime

import numpy
import tensorflow as tf
from keras import initializers

from lanmodel import embedding_util
from classifier import classifier_main as cml

from classifier import classifier_dnn_scalable as dnn_classifier
from classifier import dnn_util as util
from exp.wop import exp_wop_cml as exp_util
import pandas as pd
from exp import exp_util
from feature import text_feature_extractor as tfe
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


def run_setting(setting_file, home_dir,
                train_data_file, test_data_file,
                model_choice, #dnn - including cnn,bilstm,han; cml -svm ; fasttext-fasttext
                dataset_type:str,  #mwpd, wdc, rakuten, icecat
                dataset_text_field_mapping:dict,
                overwrite_params=None,
                embedding_format=None):
    properties = exp_util.load_properties(setting_file)

    word_weights_file = exp_util.load_setting('word_weights_file', properties, overwrite_params)
    if word_weights_file == None:
        word_weights = None
    else:
        print("using word weights to revise embedding vectors...")
        word_weights = load_word_weights(word_weights_file)

    print("loading dataset...")
    if dataset_type=="mwpd":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonMPWD(train_data_file, test_data_file)
    elif dataset_type=="rakuten":
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_csvRakuten(train_data_file, test_data_file, delimiter="\t")
    elif dataset_type=="icecat":
        pass
    else:#wdc
        df, train_size, test_size = exp_util. \
            load_and_merge_train_test_data_jsonWDC(train_data_file, test_data_file)

    #numpy.nan_to_num(df)

    class_fieldname = exp_util.load_setting("class_fieldname", properties, overwrite_params)
    class_col = dataset_text_field_mapping[class_fieldname]
    y = df[:, class_col]

    # this is the folder to save output to
    outfolder = home_dir + exp_util.load_setting("output_folder", properties, overwrite_params)

    print("\n" + str(datetime.datetime.now()))
    print("loading embedding models...")
    # this the Gensim compatible embedding file
    dnn_embedding_file=exp_util.load_setting("embedding_file", properties,
                                                          overwrite_params)
    if dnn_embedding_file is not None:
        dnn_embedding_file = home_dir + dnn_embedding_file  # "H:/Python/glove.6B/glove.840B.300d.bin.gensim"
    # print("embedding file is========="+dnn_embedding_file)
    if embedding_format == 'none':
        emb_model=None
    else:
        emb_model = embedding_util.load_emb_model(embedding_format, dnn_embedding_file)

    if model_choice == 'dnn':
        run_dnn_models(properties, df, y, train_size, class_col, outfolder, emb_model, embedding_format,
                       word_weights,dataset_text_field_mapping)
    elif model_choice=='cml':
        run_cml_models(setting_file,
                       properties, df, y, train_size, class_col, outfolder, dnn_embedding_file,
                       emb_model, embedding_format,
                       dataset_text_field_mapping)
    else:
        run_fasttext_model(setting_file,properties, df, y, train_size, class_col, outfolder,
                           dnn_embedding_file, dataset_text_field_mapping)


def run_dnn_models(properties: dict, df: numpy.ndarray, y,
                   train_size: int, class_col: int,
                   out_folder: str, embeddingmodel, embeddingformat,
                   word_weights: list,
                   text_field_mapping:dict):
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
        for x in exp_util.load_setting("text_fieldnames", properties, overwrite_params).split("|"):
            config = x.split(",")
            map = {}

            map["text_col"] = text_field_mapping[config[0]]
            map["text_length"] = int(config[1])
            map["text_dim"] = util.DNN_EMBEDDING_DIM
            input_text_info[count] = map

            # if config[1] == 'simple':
            #     dnn_branch = dnn_classifier.create_dnn_branch(map["text_length"],
            #                                                   util.DNN_EMBEDDING_DIM,
            #                                                   model_descriptor='simple'
            #                                                   )
            # else:
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

        dnn_classifier.fit_dnn_holdout(df=df,
                                       split_at_row=train_size,
                                       class_col=class_col,
                                       final_model=final_model,
                                       outfolder=out_folder,
                                       task=exp_util.describe_task(properties, overwrite_params, setting_file),
                                       model_descriptor=model_descriptor, text_norm_option=1,
                                       text_input_info=input_text_info,
                                       embedding_model=embeddingmodel,
                                       embedding_model_format=embeddingformat,
                                       word_weights=word_weights)
    print("Completed running all models on this setting file")
    print(datetime.datetime.now())


# traditional machine learning
def run_cml_models(setting_file: str,
                   properties: dict, df: numpy.ndarray, y,
                   train_size: int, class_col: int,
                   out_folder: str, embeddingmodel_file:str, embeddingmodel, embeddingformat,
                   text_field_mapping:dict):
    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    input_text_info = {}
    count = 0
    for x in exp_util.load_setting("text_fieldnames", properties, overwrite_params).split("|"):
        config = x.split(",")
        map = {}
        map["text_col"] = text_field_mapping[config[0]]
        map["text_length"] = int(config[1])
        map["text_dim"] = util.DNN_EMBEDDING_DIM
        input_text_info[count] = map

        count += 1

    print("creating feature matrix")
    X_all = []
    for k, v in input_text_info.items():
        X_sub = tfe.get_aggr_embedding_vectors(df=df,
                                               text_col=v["text_col"],
                                               text_norm_option=1,
                                               aggr_option=0,
                                               emb_format=embeddingformat,
                                               emb_model=embeddingmodel,
                                               emb_dim=int(v["text_dim"]))
        X_all.append(X_sub)
    X_all = numpy.concatenate(X_all, axis=1)

    X_train=X_all[0:train_size]
    X_test=X_all[train_size:]
    y_train=y[0:train_size]
    y_test=y[train_size:]

    setting_file = setting_file[setting_file.rfind("/") + 1:]

    models = ["svm_l"]
    for model_name in models:
        identifier=model_name+"|"+embeddingmodel_file[embeddingmodel_file.rfind("/")+1:]
        print("\tML model and embedding=" + model_name)
        print("fitting model...")

        cls = cml.Classifer(setting_file, identifier, X_train, y_train, out_folder,
                            categorical_targets=y,
                            nfold=None, algorithms=[model_name])
        trained_model=cls.run()[model_name]
        cls.eval_holdout(trained_model,model_name,X_test, y_test)

    print("Completed running all models on this setting file")
    print(datetime.datetime.now())


def run_fasttext_model(setting_file: str,
                   properties: dict, df: numpy.ndarray, y,
                   train_size: int, class_col: int,
                   outfolder: str, dnn_embedding_file,
                   text_field_mapping:dict
                   ):

    # this is the folder to save output to

    print("\n" + str(datetime.datetime.now()))

    target_classes = len(set(y))
    print("\ttotal classes=" + str(target_classes))
    print('[STARTED] running settings with label=' + exp_util.load_setting("label", properties, overwrite_params))

    print("fitting model...")

    input_text_info = {}
    count = 0
    for x in exp_util.load_setting("text_fieldnames", properties, overwrite_params).split("|"):
        config = x.split(",")
        map = {}
        map["text_col"] = text_field_mapping[config[0]]
        map["text_length"] = int(config[1])
        map["text_dim"] = util.DNN_EMBEDDING_DIM
        input_text_info[count] = map

        count += 1


    dnn_classifier.fit_fasttext_holdout(df=df,
                                split_at_row=train_size,
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

    mwpd_fieldname_to_colindex_map = {
        'ID': 0,
        'Name': 1,
        'Description': 2,
        'CategoryText': 3,
        'URL': 4,
        'lvl1': 5,
        'lvl2': 6,
        'lvl3': 7,
    }
    ##    ID, Name, Desc, Brand, Manufacturer, URL, lvl1
    wdc_fieldname_to_colindex_map = {
        'ID': 0,
        'Name': 1,
        'Desc': 2,
        'Brand': 3,
        'Manufacturer': 4,
        'URL': 5,
        'lvl1': 6
    }

    rakuten_fieldname_to_colindex_map = {
        'Name': 0,
        'lvl1': 1
    }


    if sys.argv[5]=='mwpd':
        text_field_mapping=mwpd_fieldname_to_colindex_map
    elif sys.argv[5]=='wdc':
        text_field_mapping=wdc_fieldname_to_colindex_map
    elif sys.argv[5]=='rakuten':
        text_field_mapping = rakuten_fieldname_to_colindex_map
    else:
        pass

    setting_file = sys.argv[1]

    run_setting(setting_file,
                sys.argv[2],  # tmp folder
                sys.argv[3],  # train
                sys.argv[4],  # test
                dataset_type=sys.argv[5],
                dataset_text_field_mapping=text_field_mapping,
                model_choice=sys.argv[6],
                overwrite_params=overwrite_params, embedding_format=sys.argv[7])

'''
/home/zz/Work/wop/input/dnn_holdout/mwpd/n+d+c/gslvl1_n+d+c.txt
/home/zz/Work
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json
mwpd
cml
gensim

/home/zz/Work/wop/input/dnn_holdout/mwpd/n+d+c/gslvl2_n+d+c.txt
/home/zz/Work
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json
mwpd
cml
word2vec-False
embedding_file=/data/embeddings/wop/w2v_cbow_skip.txt
'''
#word2vec-False

'''
/home/zz/Work/wop/input/dnn_holdout/wdcgs/dnn_n/gslvl1_name.txt
/home/zz/Work
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_test.json
wdc
cml
gensim
'''

'''
/home/zz/Work/wop/input/dnn_holdout/rakuten/dnn_n/gslvl1_name.txt
/home/zz/Work
/home/zz/Work/data/Rakuten/rdc-catalog-gold-small1.tsv
/home/zz/Work/data/Rakuten/rdc-catalog-gold-small2.tsv
rakuten
cml
gensim
'''


'''
/home/zz/Work/wop/input/dnn_holdout/mwpd/n+d+c/gslvl1_n+d+c.txt
/home/zz/Work
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/train.json
/home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/swc2020/test.json
mwpd
fasttext
none
embedding_file=/data/embeddings/wop/w2v_desc_cbow.txt
'''