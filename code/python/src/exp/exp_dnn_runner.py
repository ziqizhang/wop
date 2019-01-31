import sys
from deprecated import annotator as ann

import datetime
from numpy.random import seed
seed(1)

from exp import feature_creator as fc
from classifier import classifier_main as cm
import pandas as pd


if __name__ == "__main__":
    # this is the file pointing to the CSV file containing the profiles to classify, and the profile texts from which we need to extract features
    training_text_features = sys.argv[1]  # [your path]/wop/data/ml/training_text_features.csv
    # this is the folder containing other gazetteer based features that are already pre-extracted
    training_other_features = sys.argv[2]  # [your path]/wop/data/ml/training_other_features/gazetteer/dict1_match_to_profile.csv

    # this is the folder to save output to
    outfolder = sys.argv[3]

    # this the Gensim compatible embedding file
    dnn_embedding_file = "/home/zz/Work/data/glove.840B.300d.bin.gensim"

    tweets_exta=None
    #this line is used if we have 'pre-trained' dnn model and want to load it to use. i.e., we are not doing n-fold validation
    #for now, ignore this
    if len(sys.argv)>4:
        tweets_exta=ann.generate_extra_data_for_embeddingvocab(sys.argv[4],sys.argv[5])

    n_fold = 10

    # a dictionary holding different combinations of features to test
    datafeatures = {}
    datafeatures["feature_combo1"] = (training_text_features,
                                    training_other_features)

    #in order to test different DNN architectures, I implemented a parser that analyses a string following
    #specific syntax, creates different architectures. This one here takes word embedding, pass it to 3
    #cnn layer then concatenate the output by max pooling finally into a softmax
    #
    #So you can add mulitple descriptors in to a list, and the program will go through each model structure, apply them
    # to the same dataset for experiments
    #
    #the descriptor is passed as a param to 'Classifer', which parses the string to create a model
    #see 'classifier_learn.py - learn_dnn method for details
    model_descriptors=["cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]#,
                       # "scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv",
                       # "scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"]

    ######## dnn #######
    for k, v in datafeatures.items():
        print(datetime.datetime.now())

        csv_training_text_data = v[0]
        csv_training_other_feaures = v[1]
        print(csv_training_text_data)

        for model_descriptor in model_descriptors:
            print("\t"+model_descriptor)

            #SETTING0 dnn applied to profile only
            X, y = fc.create_features_text(csv_training_text_data)
            df = pd.read_csv(csv_training_text_data, header=0, delimiter=",", quoting=0).as_matrix()
            df.astype(str)
            profiles = df[:, 22]
            profiles = ["" if type(x) is float else x for x in profiles]
            cls = cm.Classifer("stakeholdercls", "_dnn_text_", X, y, outfolder,
                               categorical_targets=6, algorithms=["dnn"], nfold=n_fold,
                               text_data=profiles, dnn_embedding_file=dnn_embedding_file,
                               dnn_descriptor=model_descriptor)
            cls.run()

            print(datetime.datetime.now())
            #X would be the 'metafeature' to pass to the dnn model. Note it MUST NOT contain text and should be
            #ready-to-use features
            X, y = fc.create_features_gazetteer(csv_training_text_data, csv_training_other_feaures)
            df = pd.read_csv(csv_training_text_data, header=0, delimiter=",", quoting=0).as_matrix()
            df.astype(str)
            profiles = df[:, 22]
            profiles = ["" if type(x) is float else x for x in profiles]
            cls = cm.Classifer("stakeholdercls", "_dnn_text+other_", X, y, outfolder,
                               categorical_targets=6, algorithms=["dnn"],nfold=n_fold,
                               text_data=profiles, dnn_embedding_file=dnn_embedding_file,
                               dnn_descriptor=model_descriptor,
                               dnn_text_data_extra_for_embedding_vcab=tweets_exta)
            cls.run()
            # #
            # print(datetime.datetime.now())
            #X, y = fc.create_text_and_numeric_and_autodictext(csv_basic_feature, csv_other_feature)
            # df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
            # df.astype(str)
            # profiles = df[:, 22]
            # profiles = ["" if type(x) is float else x for x in profiles]
            # cls = cm.Classifer("stakeholdercls", "_dnn_text+numeric+autodictext_", X, y, outfolder,
            #                    categorical_targets=6, algorithms=["dnn"],nfold=n_fold,
            #                    text_data=profiles, dnn_embedding_file=dnn_embedding_file,
            #                    dnn_descriptor=model_descriptor)
            # cls.run()

