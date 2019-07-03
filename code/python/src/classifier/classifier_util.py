import pickle

import datetime

from sklearn.metrics import classification_report, accuracy_score
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels
import pandas as pd

def load_classifier_model(classifier_pickled=None):
    if classifier_pickled:
        with open(classifier_pickled, 'rb') as model:
            classifier = pickle.load(model)
        return classifier

def outputPredictions(pred, truth, model_name, task, outfolder):
    filename = os.path.join(outfolder, "predictions-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for p, t in zip(pred, truth):
        if p==t:
            line=str(p)+","+str(t)+",ok\n"
            file.write(line)
        else:
            line=str(p)+","+str(t)+",wrong\n"
            file.write(line)
    file.close()

def saveOutput(prediction, model_name, task,outfolder):
    filename = os.path.join(outfolder, "prediction-%s-%s.csv" % (model_name, task))
    file = open(filename, "w")
    for entry in prediction:
        file.write(str(entry)+"\n")
    file.close()


def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision, recall, f1, support\n"
    for i, label in enumerate(labels):
        string= string+target_names[i]+","
        for v in (p[i], r[i], f1[i]):
            string = string+"{0:0.{1}f}".format(v, digits)+","
        string = string+"{0}".format(s[i])+"\n"
        #values += ["{0}".format(s[i])]
        #report += fmt % tuple(values)

    #average
    string+="mac avg,"
    for v in (np.average(p),
              np.average(r),
              np.average(f1)):
        string += "{0:0.{1}f}".format(v, digits)+","
    string += '{0}'.format(np.sum(s))
    return string

def save_scores(predictions, gs, model_name, task_name,
                identifier, digits, outfolder):
    outputPredictions(predictions, gs, model_name, task_name, outfolder)
    filename = os.path.join(outfolder, "%s-%s.csv" % (model_name, task_name))
    file = open(filename, "a+")
    file.write(identifier)

    file.write("N-fold results:\n")
    labels = unique_labels(gs, predictions)
    target_names = ['%s' % l for l in labels]
    p, r, f1, s = precision_recall_fscore_support(gs, predictions,
                                                  labels=labels)
    acc=accuracy_score(gs, predictions)
    mac_prf_line=prepare_score_string(p,r,f1,s,labels,target_names,digits)

    prf_mac_weighted=precision_recall_fscore_support(gs, predictions,
                                                     average='weighted')
    line = mac_prf_line + "\nmacro avg weighted," + \
           str(prf_mac_weighted[0]) + "," + str(prf_mac_weighted[1]) + "," + \
           str(prf_mac_weighted[2]) + "," + str(prf_mac_weighted[3])

    prf = precision_recall_fscore_support(gs, predictions,
                                          average='micro')
    line=line+"\nmicro avg,"+str(prf[0])+","+str(prf[1])+","+\
         str(prf[2])+","+str(prf[3])
    file.write(line)
    file.write("\naccuracy on this run="+str(acc)+"\n\n")

    file.close()


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def save_classifier_model(model, outfile):
    try:
        if model:
            with open(outfile, 'wb') as model_file:
                pickle.dump(model, model_file)
    except AttributeError:
        print("Saving model failed. Perhaps not supported.")


def print_eval_report(best_params, cv_score, prediction_dev,
                      time_predict_dev,
                      time_train, y_test):
    print("CV score [%s]; best params: [%s]" %
          (cv_score, best_params))
    print("\nTraining time: %fs; "
          "Prediction time for 'dev': %fs;" %
          (time_train, time_predict_dev))
    print("\n %fs fold cross validation score:" % cv_score)
    print("\n test set result:")
    print("\n" + classification_report(y_test, prediction_dev))


def timestamped_print(msg):
    ts = str(datetime.datetime.now())
    print(ts + " :: " + msg)

#check to ensure data does not contain NaN values
def validate_training_set(training_set):
    """
    validate training data set (i.e., X) before scaling, PCA, etc.
    :param training_set: training set, test data
    :return:
    """
    # print("np any isnan(X): ", np.any(np.isnan(training_set)))
    # print("np all isfinite: ", np.all(np.isfinite(training_set)))
    # check any NaN row
    print("Training set info: " + str(training_set.shape))
    df=pd.DataFrame(training_set)
    row_i = 0
    for i in training_set:
        row_i += 1
        if np.any(np.isnan(i)):
            print("ERROR: [", row_i, "] is nan: ")
            print(i)


def feature_scaling_mean_std(feature_set):
    if feature_set is not None:
        scaler = StandardScaler(with_mean=True, with_std=True)
        return scaler.fit_transform(feature_set)
    else:
        return feature_set


def feature_scaling_min_max(feature_set):
    """
    Input X must be non-negative for multinomial Naive Bayes model
    :param feature_set:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(feature_set)


def clean_training_data(inFile, outFile):
    with open(inFile) as f:
        content = f.readlines()

    with open(outFile, 'w') as the_file:
        count=0
        for l in content:
            if count==0:
                count+=1
                the_file.write(l)
                continue

            if count>0 and l.startswith("INDEX"):
                continue
            the_file.write(l)



#clean_training_data("data/raw/training_per_raw.csv","data/raw/training_per_raw_c.csv")