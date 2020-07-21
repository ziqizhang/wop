import csv
import random

import datetime
import fasttext
import numpy
import numpy as np
import os

from keras.layers import Dense, concatenate
from keras.models import clone_model
from keras.utils import plot_model
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from classifier import classifier_util as util

from util import nlp
from keras import Model, Input
from classifier import dnn_util as dmc
from classifier import classifier_learn as cl

GLOBAL_embedding_randomized_vectors = {}
GLOBAL_embedding_random_candidates = []  # list of word indexes in the embedding model to be randomly chosen
GLOBAL_embedding_words_matched = set()
GLOBAL_embedding_vocab_indexes = []


#word_weights: if provided, will be used to weigh the embedding vectors. When a word is not found, weight of 0.5
#is applied. Otherwise, the words are ranked, then the weight is taken as (total_words - rank)^2/total_words^2
# this can not be used with fast text, which requires a file path
def text_to_vector_fasttext(text, ft_model, text_length, dim, text_norm_option, word_weigts:list=None):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = nlp.normalize(text)
    words = nlp.tokenize(text, text_norm_option)
    window = words[-text_length:]

    x = np.zeros((text_length, dim))

    for i, word in enumerate(window):
        vec = ft_model.get_word_vector(word).astype('float32')
        weight=get_word_weight(word_weigts,word)
        vec=vec*weight
        x[i, :]=vec


    return x


def text_to_vector_gensim(text, model, text_length, dim, text_norm_option,word_weigts:list=None):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = nlp.normalize(text)
    words = nlp.tokenize(text, text_norm_option)
    window = words[-text_length:]

    x = np.zeros((text_length, dim))

    random_candidates = []  # list of word indexes in the embedding model to be randomly chosen
    words_matched = set()  # track words that already found in the embedding model and whose vectors are already used

    for i, word in enumerate(window):
        weight = get_word_weight(word_weigts, word)
        is_in_model = False
        if word in model.wv.vocab.keys():
            is_in_model = True
            vec = model.wv[word]
            vec=vec*weight
            x[i, :] = vec
            words_matched.add(word)

        if not is_in_model:
            if word in GLOBAL_embedding_randomized_vectors.keys():
                vec = GLOBAL_embedding_randomized_vectors[word]
            else:
                if len(GLOBAL_embedding_vocab_indexes) == 0:
                    for n in range(0, len(model.wv.vocab.keys())):
                        GLOBAL_embedding_vocab_indexes.append(n)
                    random.Random(4).shuffle(GLOBAL_embedding_vocab_indexes)

                while (True):
                    index = GLOBAL_embedding_vocab_indexes.pop()
                    word = model.wv.index2word[index]
                    if not word in words_matched:
                        words_matched.add(word)
                        break

                vec = model.wv[word]
                GLOBAL_embedding_randomized_vectors[word] = vec

            vec = vec * weight
            x[i, :] = vec
    return x

def get_word_weight(word_weigts:list, word:str):
    if word_weigts==None:
        return 1.0
    if word in word_weigts:
        idx=word_weigts.index(word)+1
        rank= len(word_weigts)-idx
        weight = rank*rank/(len(word_weigts)*len(word_weigts))
        return weight
    else:
        return 0.5

def concate_text(row: list, col_indexes):
    text = ""
    for c in str(col_indexes).split("-"):
        text += row[int(c)] + " "
    return text.strip()


# text_col_info: an ordered list of "[index, text length, dim]"
def data_generator(df, class_col: int, batch_size, text_norm_option, classes: dict, embedding_model,
                   text_input_info: dict, embedding_format, shuffle=True, word_weights:list=None):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0  # Counter inside the current batch vector
    batch_x_multinput = []
    for k in text_input_info.keys():
        batch_x_multinput.append([])

    batch_y = None

    while True:  # Loop forever
        if shuffle:
            numpy.random.shuffle(df)  # Shuffle df each epoch

        for i in range(len(df)):
            row = df[i]
            if batch_y is None:
                batch_y = np.zeros((batch_size, len(classes)), dtype='int')
                for b in range(
                        len(text_input_info)):  # initiate each channel of input as a text_length x text_dim matrix
                    info = text_input_info[b]
                    batch_x_multinput[b] = np.zeros((batch_size, info["text_length"], info["text_dim"]),
                                                    dtype='float32')

            for b in range(len(text_input_info)):
                info = text_input_info[b]

                if embedding_format == 'fasttext':
                    batch_x_multinput[b][batch_i] = text_to_vector_fasttext(concate_text(row, info["text_col"]),
                                                                            embedding_model,
                                                                            info["text_length"], info["text_dim"],
                                                                            text_norm_option,word_weights)
                else:
                    batch_x_multinput[b][batch_i] = text_to_vector_gensim(concate_text(row, info["text_col"]),
                                                                          embedding_model,
                                                                          info["text_length"], info["text_dim"],
                                                                          text_norm_option,word_weights)
            # create the label vector
            cls = row[class_col]
            cls_index = classes[cls]
            batch_y[batch_i][cls_index] = 1
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                # 'print("batch")
                yield batch_x_multinput, batch_y
                batch_x_multinput = []
                for k in text_input_info.keys():
                    batch_x_multinput.append([])
                batch_y = None
                batch_i = 0

def create_dnn_branch(
        input_text_sentence_length,
        input_text_word_embedding_dim,
        model_descriptor):
    print("\t== Creating DNN branch (text features)...")  # create model

    # now let's assemble the model based ont the descriptor
    model_text_input_shape = Input(shape=(input_text_sentence_length, input_text_word_embedding_dim))  # model input

    model_text = dmc.create_submodel_text(
        # this parses 'model_descriptor' and takes the text-based features as input to the model
        input_layer=model_text_input_shape, model_descriptor=model_descriptor)

    # returns the dnn model, the dnn input shape, and the actual input that should match the shape
    return model_text, model_text_input_shape


'''
This method fits a dnn model (cnn, bilstm or han) using data generator to feed batches, with n-fold validation
https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings
'''


def fit_dnn(df: DataFrame, nfold: int, class_col: int,
            final_model: Model, outfolder: str,
            task: str, model_descriptor: str,
            text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format,
            word_weights:list=None):
    encoder = LabelBinarizer()
    y = df[:, class_col]

    y_int = encoder.fit_transform(y)
    y_label_lookup = dict()
    y_label_lookup_inverse = dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l] = index

    model_file = os.path.join(outfolder, "ann-%s.m" % task)
    model_copies = []
    for i in range(nfold):
        model_copy = clone_model(final_model)
        model_copy.set_weights(final_model.get_weights())
        model_copies.append(model_copy)

    kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=cl.RANDOM_STATE)
    splits = list(enumerate(kfold.split(df, y_int.argmax(1))))

    nfold_predictions = dict()
    for k in range(0, len(splits)):
        print("\tnfold=" + str(k))
        nfold_model = model_copies[k]
        nfold_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        X_train_index = splits[k][1][0]
        X_test_index = splits[k][1][1]

        X_train_merge_ = df[X_train_index]
        X_test_merge_ = df[X_test_index]
        y_train_ = y_int[X_train_index]
        y_test_ = y_int[X_test_index]

        # df, batch_size, text_norm_option, classes, ft_model, text_col_info:list
        training_generator = data_generator(df=X_train_merge_,
                                            class_col=class_col,
                                            classes=y_label_lookup_inverse,
                                            batch_size=dmc.DNN_BATCH_SIZE,
                                            text_norm_option=text_norm_option,
                                            embedding_model=embedding_model,
                                            text_input_info=text_input_info,
                                            embedding_format=embedding_model_format,
                                            word_weights=word_weights)

        training_steps_per_epoch = round(len(X_train_merge_) / dmc.DNN_BATCH_SIZE)

        nfold_model.fit_generator(training_generator, steps_per_epoch=training_steps_per_epoch,
                                  epochs=dmc.DNN_EPOCHES)

        test_generator = data_generator(df=X_test_merge_,
                                        class_col=class_col,
                                        classes=y_label_lookup_inverse,
                                        batch_size=len(X_test_merge_),
                                        text_norm_option=text_norm_option,
                                        embedding_model=embedding_model,
                                        text_input_info=text_input_info,
                                        embedding_format=embedding_model_format,shuffle=False,
                                        word_weights=word_weights)
        prediction_prob = nfold_model.predict_generator(test_generator, steps=1)

        # evaluate the model
        #
        predictions = prediction_prob.argmax(axis=-1)

        for i, l in zip(X_test_index, predictions):
            nfold_predictions[i] = l

        del nfold_model

    indexes = sorted(list(nfold_predictions.keys()))
    predicted_labels = []
    for i in indexes:
        predicted_labels.append(nfold_predictions[i])
    util.save_scores(predicted_labels, y_int.argmax(1), "dnn", task, model_descriptor, 3,
                     outfolder)


'''
This method fits a dnn model (cnn, bilstm or han) using data generator to feed batches, with holdout validation

df: DataFrame, nfold: int,class_col:int,
            final_model: Model, outfolder: str,
            task: str, model_descriptor: str,
            text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format
'''


def fit_dnn_holdout(df: DataFrame, split_at_row: int, class_col: int,
                    final_model: Model, outfolder: str,
                    task: str, model_descriptor: str,
                    text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format,
                    word_weights: list = None):
    encoder = LabelBinarizer()
    y = df[:, class_col]
    print("\ttotal y rows="+str(len(y))+" with unique values="+str(len(set(y))))
    print("\tencoding y labels..."+str(datetime.datetime.now()))
    y_int = encoder.fit_transform(y)

    print("\tcreating y labels dictionary..." + str(datetime.datetime.now()))
    y_label_lookup = dict()
    y_label_lookup_inverse = dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l] = index

    model_file = os.path.join(outfolder, "ann-%s.m" % task)

    print("\tspliting to train/test..." + str(datetime.datetime.now()))
    df_train = df[0:split_at_row]
    df_test = df[split_at_row:]

    # nfold_predictions = dict()

    print("\ttraining..."+str(datetime.datetime.now()))
    final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_generator = data_generator(df=df_train,
                                        class_col=class_col,
                                        classes=y_label_lookup_inverse,
                                        batch_size=dmc.DNN_BATCH_SIZE,
                                        text_norm_option=text_norm_option,
                                        embedding_model=embedding_model,
                                        text_input_info=text_input_info,
                                        embedding_format=embedding_model_format,
                                        word_weights=word_weights)

    training_steps_per_epoch = round(len(df_train) / dmc.DNN_BATCH_SIZE)

    final_model.fit_generator(training_generator, steps_per_epoch=training_steps_per_epoch,
                              epochs=dmc.DNN_EPOCHES)

    print("\ttesting...")
    test_generator = data_generator(df=df_test,
                                    class_col=class_col,
                                    classes=y_label_lookup_inverse,
                                    batch_size=len(df_test),
                                    text_norm_option=text_norm_option,
                                    embedding_model=embedding_model,
                                    text_input_info=text_input_info,
                                    embedding_format=embedding_model_format, shuffle=False,
                                    word_weights=word_weights)
    prediction_prob = final_model.predict_generator(test_generator, steps=1)

    # evaluate the model
    #
    predictions = prediction_prob.argmax(axis=-1)

    # evaluate the model
    util.save_scores(predictions, y_int[split_at_row:,:].argmax(1), "dnn", task, model_descriptor, 3,
                     outfolder)


'''
this model fits a fasttext model in nfold. the embedding model must meets the fasttext format

df: DataFrame, nfold: int,class_col:int,
            final_model: Model, outfolder: str,
            task: str, model_descriptor: str,
            text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format
'''


def fit_fasttext(df: DataFrame, nfold: int, class_col: int,
                 outfolder: str,
                 task: str,
                 text_norm_option: int, text_input_info: dict, embedding_file: str):
    # X, y, embedding_file, nfold, outfolder: str, task: str):
    print("\t running fasttext using embedding file="+str(embedding_file))
    encoder = LabelBinarizer()
    y = df[:, class_col]

    y_int = encoder.fit_transform(y)
    y_label_lookup = dict()
    y_label_lookup_inverse = dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l] = index
        # print(l+","+str(index))

    X = []
    text_length = 0
    index = 0
    for row in df:
        text = ""
        for b in range(len(text_input_info)):
            info = text_input_info[b]
            text += concate_text(row, info["text_col"]) + " "
            text_length += int(info["text_length"])
        text = nlp.normalize(text)
        words = nlp.tokenize(text, text_norm_option)
        text = " ".join(words).strip()
        X.append([text])
        index += 1
    X = numpy.asarray(X, dtype=str)

    # perform n-fold validation (we cant use scikit-learn's wrapper as we used Keras functional api above
    kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=cl.RANDOM_STATE)
    splits = list(enumerate(kfold.split(X, y_int.argmax(1))))

    nfold_predictions = dict()
    for k in range(0, len(splits)):
        print("\tnfold=" + str(k))

        # Fit the model
        X_train_index = splits[k][1][0]
        X_test_index = splits[k][1][1]

        X_train_ = X[X_train_index]
        y_train_ = y[X_train_index]
        X_test_ = X[X_test_index]
        y_test_ = y[X_test_index]

        # prepare fasttext data
        fasttext_train = outfolder + "/fasttext_train.tsv"
        with open(fasttext_train, mode='w') as outfile:
            csvwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(X_train_)):
                label = y_train_[i]
                text = X_train_[i][0]
                csvwriter.writerow(["__label__" + label.replace(" ", "|"), text])

        # fasttext_test = outfolder + "/fasttext_test.tsv"
        # with open(fasttext_test, mode='w') as outfile:
        #     csvwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     for i in range(len(X_test_)):
        #         label = y_test_[i]
        #         text = X_test_[i][0]
        #         csvwriter.writerow(["__label__" + label, text])

        # -dim 300 -minn 4 -maxn 10 -wordNgrams 3 -neg 10 -loss ns -epoch 3000 -thread 30
        if embedding_file is not None:
            model = fasttext.train_supervised(input=fasttext_train,
                                          minn=4, maxn=10, wordNgrams=3,
                                          neg=10, loss='ns', epoch=3000,
                                          thread=30,
                                          dim=dmc.DNN_EMBEDDING_DIM,
                                          pretrainedVectors=embedding_file)
        else:
            model = fasttext.train_supervised(input=fasttext_train,
                                              minn=4, maxn=10, wordNgrams=3,
                                              neg=10, loss='ns', epoch=3000,
                                              thread=30,
                                              dim=dmc.DNN_EMBEDDING_DIM)

        # evaluate the model
        X_test_as_list = []
        for row in X_test_:
            X_test_as_list.append(row[0])
        predictions = model.predict(X_test_as_list)[0]

        for i in range(len(X_test_index)):
            index = X_test_index[i]
            label = predictions[i][0]
            l = label[9:]
            l = l.replace("|", " ")
            nfold_predictions[index] = y_label_lookup_inverse[l]

    indexes = sorted(list(nfold_predictions.keys()))
    predicted_labels = []
    for i in indexes:
        predicted_labels.append(nfold_predictions[i])

    util.save_scores(predicted_labels, y_int.argmax(1), "dnn", task, "_fasttext_", 3,
                     outfolder)


def fit_fasttext_holdout(df: DataFrame, split_at_row: int, class_col: int,
                         outfolder: str,
                         task: str,
                         text_norm_option: int, text_input_info: dict, embedding_file: str):
    # X, y, embedding_file, nfold, outfolder: str, task: str):

    encoder = LabelBinarizer()
    y = df[:, class_col]

    y_int = encoder.fit_transform(y)
    y_label_lookup = dict()
    y_label_lookup_inverse = dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l] = index
        # print(l+","+str(index))

    X = []
    text_length = 0
    index = 0
    for row in df:
        text = ""
        for b in range(len(text_input_info)):
            info = text_input_info[b]
            t= concate_text(row, info["text_col"])
            t=nlp.normalize(t)
            text_length += int(info["text_length"])
            text+=t+" "
        words = nlp.tokenize(text, text_norm_option)
        text = " ".join(words).strip()
        X.append([text])
        index += 1
    X = numpy.asarray(X, dtype=str)

    # perform n-fold validation (we cant use scikit-learn's wrapper as we used Keras functional api above

    X_train_ = X[0:split_at_row]
    y_train_ = y[0:split_at_row]
    X_test_ = X[split_at_row:]
    y_test_ = y[split_at_row:]

    # prepare fasttext data
    fasttext_train = outfolder + "/fasttext_train.tsv"
    with open(fasttext_train, mode='w') as outfile:
        csvwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X_train_)):
            label = y_train_[i]
            text = X_train_[i][0]
            csvwriter.writerow(["__label__" + label.replace(" ", "|"), text])

        # fasttext_test = outfolder + "/fasttext_test.tsv"
        # with open(fasttext_test, mode='w') as outfile:
        #     csvwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     for i in range(len(X_test_)):
        #         label = y_test_[i]
        #         text = X_test_[i][0]
        #         csvwriter.writerow(["__label__" + label, text])

        # -dim 300 -minn 4 -maxn 10 -wordNgrams 3 -neg 10 -loss ns -epoch 3000 -thread 30
    if embedding_file is not None and embedding_file.lower()!='none':
        model = fasttext.train_supervised(input=fasttext_train,
                                      minn=4, maxn=10, wordNgrams=3,
                                      neg=10, loss='ns', epoch=3000,
                                      thread=30,
                                      dim=dmc.DNN_EMBEDDING_DIM,
                                      pretrainedVectors=embedding_file)
    else:
        model = fasttext.train_supervised(input=fasttext_train,
                                          minn=4, maxn=10, wordNgrams=3,
                                          neg=10, loss='ns', epoch=3000,
                                          thread=30,
                                          dim=dmc.DNN_EMBEDDING_DIM)
    # evaluate the model

    X_test_as_list = []
    for row in X_test_:
        X_test_as_list.append(row[0])
    predictions = model.predict(X_test_as_list)[0]

    predicted_labels = []
    for i in predictions:
        label = i[0]
        l = label[9:]
        l = l.replace("|", " ")
        predicted_labels.append(y_label_lookup_inverse[l])

    util.save_scores(predicted_labels, y_int[split_at_row:,:].argmax(1), "dnn", task, "_fasttext_", 3,
                     outfolder)


def merge_dnn_branch(branches: list, input_shapes: list, prediction_targets: int):
    if len(branches) > 1:
        merge = concatenate(branches)
    else:
        merge = branches[0]
    full = Dense(600)(merge)
    final = Dense(prediction_targets, activation="softmax")(full)
    model = Model(inputs=input_shapes, outputs=final)
    model.summary()

    # this prints the model architecture diagram to a file, so you can check that it looks right
    plot_model(model, to_file="model.png")
    return model