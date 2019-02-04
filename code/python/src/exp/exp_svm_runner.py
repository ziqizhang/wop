import sys

import datetime

from exp import feature_creator as fc
from classifier import classifier_main as cm

if __name__ == "__main__":
    #this is the file pointing to the CSV file containing the profiles to classify, and the profile texts from which we need to extract features
    training_text_features=sys.argv[1]# [your path]/wop/data/ml/training_text_features.csv
    #this is the folder containing other gazetteer based features that are already pre-extracted
    training_other_features=sys.argv[2] #[your path]/wop/data/ml/training_other_features

    #this is the folder to save output to
    outfolder=sys.argv[3] #[your path]/wop/output/classifier

    #n fold validation
    n_fold=10


    #a dictionary holding different combinations of features to test
    datafeatures={}
    datafeatures["feature_combo1"] = (training_text_features,
                                    training_other_features)

    ######## svm, no pca #######
    for k,v in datafeatures.items():
        print(datetime.datetime.now())
        csv_basic_feature=v[0]
        csv_other_feature=v[1]
        #setting 1: use profile text feature only
        # X, y = fc.create_features_text(csv_basic_feature)
        # # the Classifier object creates different algorithms depending on the params passed. See comments inside for details
        # cls = cm.Classifer(k+"stakeholdercls", "_profiletext_", X, y, outfolder,
        #                     categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        # cls.run()
        #
        #
        # #setting 2: use profile text and other features
        # print(datetime.datetime.now())
        # X, y = fc.create_features_text_and_other(csv_basic_feature, csv_other_feature)
        # cls = cm.Classifer(k+"stakeholdercls", "_profiletext+other_", X, y, outfolder,
        #                    categorical_targets=6,nfold=n_fold,algorithms=["svm_l"])
        # cls.run()


        ####### svm, pca #######
        # setting 3: use profile text feature only, but using pca
        print(datetime.datetime.now())
        X, y = fc.create_features_text(csv_basic_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_profiletext_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        cls.run()

        # setting 4: use profile text and other features, but also using pca
        print(datetime.datetime.now())
        X, y = fc.create_features_text_and_other(csv_basic_feature, csv_other_feature)
        cls = cm.Classifer(k+"stakeholdercls", "_profiletext+other_", X, y, outfolder,
                           categorical_targets=6, nfold=n_fold, algorithms=["pca-svm_l"])
        cls.run()



