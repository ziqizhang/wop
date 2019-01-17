import gensim
import keras
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelBinarizer

from classifier import classifier_util as util
import os
from classifier import dnn_util as dmc

'''this file loads a pre-trained model and classify new data. currently
for non-DNN models, only non-text features are supported.'''

def predict(model_flag, task, model_file,test_features, text_data,outfolder):
    print("start prediction stage :: data size:", len(test_features))

    label_lookup={}
    label_lookup[0]="Advocate"
    label_lookup[1] = "HPI"
    label_lookup[2] = "HPO"
    label_lookup[3] = "Other"
    label_lookup[4] = "Patient"
    label_lookup[5] = "Research"

    if model_flag.startswith("dnn"):
        M = dmc.get_word_vocab(text_data, 1, use_saved_vocab=True)
        text_based_features = M[0]
        text_based_features = sequence.pad_sequences(text_based_features,
                                                     dmc.DNN_MAX_SEQUENCE_LENGTH)

        if model_flag=="dnn":
            yaml_file = open(model_file+'.yaml', 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            model = model_from_yaml(loaded_model_yaml)#,custom_objects={"SkipConv1D": SkipConv1D})
            # load weights into new model
            model.load_weights(model_file+".h5")
            print("Loaded model from disk")

            # evaluate loaded model on test data
        else:
            model=keras.models.load_model(model_file+".h5")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        prediction_prob=model.predict([text_based_features,test_features])
        predictions = prediction_prob.argmax(axis=-1)

        # y = test_features[:, 22]
        # encoder = LabelBinarizer()
        # y_int = encoder.fit_transform(y)
        # util.save_scores(predictions, y_int.argmax(1), "dnn", task, "_test_data_", 2,
        #                  outfolder)

    else:

        ######################### SGDClassifier #######################
        if model_flag== "sgd":
            # SGD doesn't work so well with only a few samples, but is (much more) performant with larger data
            # At n_iter=1000, SGD should converge on most datasets
            print("Using SGD ...")
            model_file = os.path.join(outfolder, "sgd-classifier-%s.m" % task)

        ######################### Stochastic Logistic Regression#######################
        if model_flag== "lr":
            print("Using Stochastic Logistic Regression ...")
            model_file = os.path.join(outfolder, "stochasticLR-%s.m" % task)

        ######################### Random Forest Classifier #######################
        if model_flag== "rf":
            print("Using Random Forest ...")
            model_file = os.path.join(outfolder, "random-forest_classifier-%s.m" % task)

        ###################  liblinear SVM ##############################
        if model_flag== "svm_l":
            print("Using SVM, kernel=linear ...")
            model_file = os.path.join(outfolder, "liblinear-svm-linear-%s.m" % task)

        ##################### RBF svm #####################
        if model_flag== "svm_rbf":
            print("Using SVM, kernel=rbf ....")
            model_file = os.path.join(outfolder, "liblinear-svm-rbf-%s.m" % task)
        model = util.load_classifier_model(model_file)

        predictions = model.predict_proba(test_features)

    filename = os.path.join(outfolder, "prediction-%s-%s.csv" % (model_flag, task))
    file = open(filename, "w")

    prediction_labels=[]
    for p in predictions:
        p_label=label_lookup[p]
        file.write(p_label + "\n")
        prediction_labels.append(p_label)
    file.close()

    return prediction_labels
    #util.saveOutput()