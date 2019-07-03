'''
Ad hoc class for dealing with DNN modles that need to (quasi-)duplicate architectures for different input. For example,
when you need the same CNN structure for different text inputs, then concatenate them for the classification task.

currently only 2D input is supported (i.e., [text,words])
'''
import numpy
import os
from keras import Input, Model
from keras.layers import concatenate, Dense
from keras.models import clone_model
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from classifier import dnn_util as dmc
from classifier import classifier_learn as cl
from classifier import classifier_util as util


def create_dnn_branch_rawfeatures(
        input_data_cols: list,
        dataframe_as_matrix: numpy.ndarray
):
    print("\t== Creating DNN branch (raw features) ...")  # create model

    # now let's assemble the model based ont the descriptor
    model_input_shape = Input(shape=(len(input_data_cols),))  # model input
    model = model_input_shape
    model_input_features = numpy.ndarray(shape=(len(dataframe_as_matrix), len(input_data_cols)), dtype=float)

    col_idx = 0
    for col in input_data_cols:
        model_input_features[:, col_idx] = dataframe_as_matrix[:, int(col)]
        col_idx += 1

    # returns the dnn model, the dnn input shape, and the actual input that should match the shape
    return model, model_input_shape, model_input_features


def create_dnn_branch_textinput(
        pretrained_embedding_models,
        input_text_data,
        input_text_sentence_length,
        input_text_word_embedding_dim,
        model_descriptor,
        text_data_extra_for_embedding_vocab=None,
        embedding_trainable=False,
        embedding_mask_zero=False):
    print("\t== Creating DNN branch (text features)...")  # create model

    # process text data, index vocabulary, pad each text sentence/paragraph to a fixed length
    M = dmc.extract_vocab_and_2D_input(input_text_data, 1, sentence_length=input_text_sentence_length,
                                       tweets_extra=text_data_extra_for_embedding_vocab)
    X_train_text_feature_input = M[0]

    # create the embedding layer by mapping each input sentence sequence to embedding representations by
    # looking up its containing words in the pre-trained embedding model
    pretrained_word_matrix = dmc.build_pretrained_embedding_matrix(M[1],
                                                                   pretrained_embedding_models,
                                                                   dmc.DNN_EMBEDDING_DIM,
                                                                   2)

    # now let's assemble the model based ont the descriptor
    model_text_input_shape = Input(shape=(input_text_sentence_length,))  # model input

    model_text = dmc.create_submodel_textfeature(
        # this parses 'model_descriptor' and takes the text-based features as input to the model
        sentence_inputs_2D=model_text_input_shape,
        # it is useful to see the details of this method and try a few different options to see difference
        max_sentence_length=input_text_sentence_length,
        word_vocab_size=len(M[1]),
        word_embedding_dim=input_text_word_embedding_dim,
        word_embedding_weights=pretrained_word_matrix,
        model_option=model_descriptor,
        word_embedding_trainable=embedding_trainable,
        word_embedding_mask_zero=embedding_mask_zero)

    # returns the dnn model, the dnn input shape, and the actual input that should match the shape
    return model_text, model_text_input_shape, X_train_text_feature_input


# branches is a dictionary where key is the input, value is the model branch for that input
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


def util_count_class_instances(y_train):
    class_freq = dict()
    for i in y_train:
        if i in class_freq.keys():
            class_freq[i] = class_freq[i] + 1
        else:
            class_freq[i] = 1
    return class_freq


def util_find_isolated_classes(class_freq_data1, class_freq_data2):
    data1_only = []
    data2_only = []
    for k, v in class_freq_data1.items():
        if v > 1:
            continue
        if k not in class_freq_data2.keys():
            data1_only.append(k)

    for k, v in class_freq_data2.items():
        if v > 1:
            continue
        if k not in class_freq_data1.keys():
            data2_only.append(k)
    return data1_only, data2_only


def find_incorrect_single_instance_pred(train_only, test_only, predictions, fold_index):
    for i in range(len(predictions)):
        p = predictions[i]
        if p in train_only:
            print("\tWARNING-instance in fold=" + str(fold_index)
                  + " class=" + str(p) + " index=" + str(
                i) + " is training split only but found in prediction. This should be a false positive.")
        elif p in test_only:
            print("\tWARNING-instance in fold=" + str(fold_index)
                  + " class=" + str(p) + " index=" + str(
                i) + " is testing split only but found in prediction. This should not happen.")


def fit_dnn(inputs: list, nfold: int, y_train,
            final_model: Model, outfolder: str,
            task: str, model_descriptor: str):
    encoder = LabelBinarizer()
    y_train_int = encoder.fit_transform(y_train)
    y_train_label_lookup = dict()
    for index, l in zip(y_train_int.argmax(1), y_train):
        y_train_label_lookup[index] = l

    X_merge = numpy.concatenate(inputs,
                                axis=1)  # merge so as to create correct splits across all different feature inputs

    model_file = os.path.join(outfolder, "ann-%s.m" % task)

    model_copies = []
    for i in range(nfold):
        model_copy = clone_model(final_model)
        model_copy.set_weights(final_model.get_weights())
        model_copies.append(model_copy)

    # perform n-fold validation (we cant use scikit-learn's wrapper as we used Keras functional api above
    if nfold is not None:
        kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=cl.RANDOM_STATE)
        splits = list(enumerate(kfold.split(X_merge, y_train_int.argmax(1))))

        nfold_predictions = dict()
        for k in range(0, len(splits)):
            print("\tnfold=" + str(k))
            nfold_model = model_copies[k]
            nfold_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Fit the model
            X_train_index = splits[k][1][0]
            X_test_index = splits[k][1][1]

            X_train_merge_ = X_merge[X_train_index]
            X_test_merge_ = X_merge[X_test_index]
            y_train_ = y_train_int[X_train_index]
            y_test_ = y_train_int[X_test_index]

            separate_training_feature_inputs = []  # to contain features for training set coming from different input branches
            separate_testing_feature_inputs = []
            index_start = 0
            for feature_input in inputs:
                length = len(feature_input[0])
                index_end = index_start + length
                slice_train = X_train_merge_[:, index_start:index_end]
                slice_test = X_test_merge_[:, index_start:index_end]
                separate_training_feature_inputs.append(slice_train)
                separate_testing_feature_inputs.append(slice_test)
                index_start = index_end

            nfold_model.fit(separate_training_feature_inputs,
                            y_train_, epochs=dmc.DNN_EPOCHES, batch_size=dmc.DNN_BATCH_SIZE)
            prediction_prob = nfold_model.predict(separate_testing_feature_inputs)

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
    else:
        final_model.fit(inputs,
                        y_train_int, epochs=dmc.DNN_EPOCHES, batch_size=dmc.DNN_BATCH_SIZE, verbose=2)

        # serialize model to YAML
        model_yaml = final_model.to_yaml()
        with open(model_file + ".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
            # serialize weights to HDF5
            final_model.save_weights(model_file + ".h5")
        # util.save_classifier_model(model, model_file)


def fit_dnn_holdout(inputs: list, y_labels,
                    final_model: Model, outfolder: str,
                    task: str, model_descriptor: str, split_row=None):
    encoder = LabelBinarizer()
    y_label_int = encoder.fit_transform(y_labels)
    y_label_lookup = dict()
    for index, l in zip(y_label_int.argmax(1), y_labels):
        y_label_lookup[index] = l

    X_merge = numpy.concatenate(inputs,
                                axis=1)  # merge so as to create correct splits across all different feature inputs

    X_train= X_merge[0:split_row]
    X_test= X_merge[split_row:]
    y_train= y_label_int[0:split_row]
    y_test=y_label_int[split_row:]

    model_file = os.path.join(outfolder, "ann-%s.m" % task)

    #nfold_predictions = dict()

    print("\ttraining...")
    final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    final_model.fit(X_train,
                    y_train, epochs=dmc.DNN_EPOCHES, batch_size=dmc.DNN_BATCH_SIZE)
    print("\ttesting...")
    prediction_prob = final_model.predict(X_test)

    # evaluate the model
    #
    predictions = prediction_prob.argmax(axis=-1)
    util.save_scores(predictions, y_test.argmax(1), "dnn", task, model_descriptor, 3,
                     outfolder)
