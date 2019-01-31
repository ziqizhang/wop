import sys

import datetime

from exp import feature_creator as fc
from classifier import classifier_main as cm

if __name__ == "__main__":
    #this is the file pointing to the basic features, i.e., just the numeric values
    #msm4phi/paper2/data/training_data/basic_features.csv
    training_text_features=sys.argv[1]
    #this is the folder containing other extracted features
    training_other_features=sys.argv[2]
    #this is needed if dnn model is used
    dnn_embedding_file="/home/zz/Work/data/glove.840B.300d.bin.gensim"

    #this is the folder to save output to
    outfolder=sys.argv[3]

    #n fold validation
    n_fold=10


    #a dictionary holding different combinations of features to test
    datafeatures={}
    datafeatures["profiles_"] = (training_text_features,
                                    training_other_features)

    ######## svm, no pca #######
    for k,v in datafeatures.items():
        print(datetime.datetime.now())
        csv_basic_feature=v[0]
        csv_other_feature=v[1]
        #use profile text feature only
        X, y = fc.create_features_text(csv_basic_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_profiletext_", X, y, outfolder,
                            categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        cls.run()


        #use profile text and other features
        print(datetime.datetime.now())
        X, y = fc.create_features_text_and_other(csv_basic_feature, csv_other_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_text+other_", X, y, outfolder,
                           categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        cls.run()


        ####### svm, pca #######
        # use profile text feature only
        print(datetime.datetime.now())
        X, y = fc.create_features_text(csv_basic_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_text_only_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        cls.run()

        # use profile text and other features
        print(datetime.datetime.now())
        X, y = fc.create_features_text_and_other(csv_basic_feature, csv_other_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_text+numeric_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        cls.run()



