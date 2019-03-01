'''
Ad hoc class for dealing with DNN modles that need to (quasi-)duplicate architectures for different input. For example,
when you need the same CNN structure for different text inputs, then concatenate them for the classification task.

currently only 2D input is supported (i.e., [text,words])
'''
import numpy
import os
from keras import Input, Model
from keras.layers import concatenate, Dense
from keras.utils import plot_model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from classifier import dnn_util as dmc
from classifier import classifier_learn as cl
from classifier import classifier_util as util


def create_dnn_branch(
        pretrained_embedding_models,
        input_text_data,
        input_text_sentence_length,
        input_text_word_embedding_dim,
        model_descriptor,
        text_data_extra_for_embedding_vocab=None,
        embedding_trainable=False,
        embedding_mask_zero=False):
    print("== Perform ANN ...")  # create model

    # process text data, index vocabulary, pad each text sentence/paragraph to a fixed length
    M = dmc.extract_vocab_and_2D_input(input_text_data, 1, sentence_length=dmc.DNN_MAX_SENTENCE_LENGTH,
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

    #returns the dnn model, the dnn input shape, and the actual input that should match the shape
    return model_text, model_text_input_shape, X_train_text_feature_input


#branches is a dictionary where key is the input, value is the model branch for that input
def merge_dnn_branch(branches:list, input_shapes:list, prediction_targets:int):
    merge = concatenate(branches)
    final = Dense(prediction_targets, activation="softmax")(merge)
    model = Model(inputs=input_shapes, outputs=final)

    # this prints the model architecture diagram to a file, so you can check that it looks right
    plot_model(model, to_file="model.png")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def fit_dnn(inputs:list, nfold:int, y_train, final_model:Model,outfolder:str, task:str, model_descriptor:str):
    encoder = LabelBinarizer()
    y_train_int = encoder.fit_transform(y_train)
    X_merge = numpy.concatenate(inputs, axis=1) #merge so as to create correct splits across all different feature inputs

    model_file = os.path.join(outfolder, "ann-%s.m" % task)

    # perform n-fold validation (we cant use scikit-learn's wrapper as we used Keras functional api above
    if nfold is not None:
        kfold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=cl.RANDOM_STATE)
        splits = list(enumerate(kfold.split(X_merge, y_train_int.argmax(1))))

        nfold_predictions = dict()
        for k in range(0, len(splits)):
            # Fit the model
            X_train_index = splits[k][1][0]
            X_test_index = splits[k][1][1]

            X_train_merge_ = X_merge[X_train_index]
            X_test_merge_ = X_merge[X_test_index]
            y_train_ = y_train_int[X_train_index]

            separate_training_feature_inputs=[]
            separate_testing_feature_inputs=[]
            index_start=0
            for feature_input in inputs:
                length = len(feature_input)
                index_end=index_start+length
                slice_train = X_train_merge_[:, index_start:index_end]
                slice_test = X_test_merge_[:, index_start:index_end]
                separate_training_feature_inputs.append(slice_train)
                separate_testing_feature_inputs.append(slice_test)
                index_start=index_end


            final_model.fit(separate_training_feature_inputs,
                          y_train_, epochs=dmc.DNN_EPOCHES, batch_size=dmc.DNN_BATCH_SIZE)
            prediction_prob = final_model.predict(separate_testing_feature_inputs)

            # evaluate the model
            #
            predictions = prediction_prob.argmax(axis=-1)

            for i, l in zip(X_test_index, predictions):
                nfold_predictions[i] = l

            # self.save_classifier_model(best_estimator, ann_model_file)

        indexes = sorted(list(nfold_predictions.keys()))
        predicted_labels = []
        for i in indexes:
            predicted_labels.append(nfold_predictions[i])
        util.save_scores(predicted_labels, y_train_int.argmax(1), "dnn", task, model_descriptor, 2,
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