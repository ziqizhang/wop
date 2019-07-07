import random

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


def text_to_vector_fasttext(text, ft_model, text_length, dim, text_norm_option):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = nlp.normalize(text)
    words = nlp.tokenize(text, text_norm_option)
    window = words[-text_length:]

    x = np.zeros((text_length, dim))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x


def text_to_vector_gensim(text, model, text_length, dim, text_norm_option):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = nlp.normalize(text)
    words = nlp.tokenize(text, text_norm_option)
    window = words[-text_length:]

    x = np.zeros((text_length, dim))

    random_candidates = []  # list of word indexes in the embedding model to be randomly chosen
    words_matched = set()  # track words that already matched and whose vectors are already used

    for i, word in enumerate(window):
        is_in_model = False
        if word in model.wv.vocab.keys():
            is_in_model = True
            vec = model.wv[word]
            x[i, :] = vec
            words_matched.add(word)

        if not is_in_model:
            if word in GLOBAL_embedding_randomized_vectors.keys():
                vec = GLOBAL_embedding_randomized_vectors[word]
            else:
                if len(random_candidates) == 0:
                    for x in range(0, len(model.wv.vocab.keys())):
                        random_candidates.append(x)
                    random.Random(4).shuffle(random_candidates)

                while (True):
                    index = random_candidates.pop()
                    word = model.wv.index2word[index]
                    if not word in words_matched:
                        words_matched.add(word)
                        break

                vec = model.wv[word]
                GLOBAL_embedding_randomized_vectors[word] = vec

            x[i, :] = vec

    return x

def concate_text(row:list, col_indexes):
    text=""
    for c in col_indexes.split("-"):
        text+=row[int(c)]+" "
    return text.strip()

# text_col_info: an ordered list of "[index, text length, dim]"
def data_generator(df, class_col:int, batch_size, text_norm_option, classes:dict, embedding_model, text_input_info: dict, embedding_format):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0  # Counter inside the current batch vector
    batch_x_multinput = []
    for k in text_input_info.keys():
        batch_x_multinput.append([])

    batch_y = None

    while True:  # Loop forever
        numpy.random.shuffle(df)  # Shuffle df each epoch

        for i in range(len(df)):
            row=df[i]
            if batch_y is None:
                batch_y = np.zeros((batch_size, len(classes)), dtype='int')
                for b in range(len(text_input_info)): #initiate each channel of input as a text_length x text_dim matrix
                    info = text_input_info[b]
                    batch_x_multinput[b] = np.zeros((batch_size, info["text_length"], info["text_dim"]), dtype='float32')

            for b in range(len(text_input_info)):
                info = text_input_info[b]

                if embedding_format=='fasttext':
                    batch_x_multinput[b][batch_i] = text_to_vector_fasttext(concate_text(row, info["text_col"]),
                                                                            embedding_model,
                                                                            info["text_length"], info["text_dim"], text_norm_option)
                else:
                    batch_x_multinput[b][batch_i] = text_to_vector_gensim(concate_text(row, info["text_col"]),
                                                                          embedding_model,
                                                                          info["text_length"], info["text_dim"],
                                                                          text_norm_option)
            #create the label vector
            cls=row[class_col]
            cls_index=classes[cls]
            batch_y[batch_i][cls_index] = 1
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
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


def fit_dnn(df: DataFrame, nfold: int,class_col:int,
            final_model: Model, outfolder: str,
            task: str, model_descriptor: str,
            text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format):
    # todo: but you need to pass input_shape=(window_length, n_features) to the first layer in your NN.

    encoder = LabelBinarizer()
    y = df[:, class_col]

    y_int = encoder.fit_transform(y)
    y_label_lookup = dict()
    y_label_lookup_inverse=dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l]=index

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
                                            embedding_format=embedding_model_format)

        training_steps_per_epoch = round(len(X_train_merge_) / dmc.DNN_BATCH_SIZE)

        nfold_model.fit_generator(training_generator,steps_per_epoch=training_steps_per_epoch,
                                  epochs=dmc.DNN_EPOCHES)
        prediction_prob = nfold_model.predict_generator(X_test_merge_)

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
    util.save_scores(predicted_labels, y_train_int.argmax(1), "dnn", task, model_descriptor, 3,
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
