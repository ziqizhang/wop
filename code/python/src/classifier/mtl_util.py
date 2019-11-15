from pandas import DataFrame
from keras import Model, Input
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import clone_model
from sklearn.preprocessing import LabelBinarizer
from classifier import classifier_learn as cl
from classifier import dnn_util as dmc
from classifier import classifier_util as util
from keras.layers import Dense, concatenate
from keras.utils import plot_model
from classifier import classifier_dnn_scalable as cl_dnn
import copy


def create_mtl_layers(branches: list, input_shapes: list, prediction_targets: int, aux_targets:list):
    '''
    x = Input(shape=(n, ))
shared = Dense(32)(x)
sub1 = Dense(16)(shared)
sub2 = Dense(16)(shared)
sub3 = Dense(16)(shared)
out1 = Dense(1)(sub1)
out2 = Dense(1)(sub2)
out3 = Dense(1)(sub3)

model = Model(inputs=x, outputs=[out1, out2, out3])

    '''
    if len(branches) > 1:
        shared = concatenate(branches)
    else:
        shared = branches[0]
    main = Dense(600)(shared)
    main_out = Dense(prediction_targets, activation="softmax")(main)

    model_outputs=[main_out]

    for cls in aux_targets:
        aux = Dense(600)(shared)
        aux_out=Dense(cls, activation="softmax")(aux)
        model_outputs.append(aux_out)

    model = Model(inputs=input_shapes, outputs=model_outputs)
    model.summary()

    # this prints the model architecture diagram to a file, so you can check that it looks right
    plot_model(model, to_file="model.png")
    return model

def encode_labels(df: DataFrame, class_col:int):
    encoder = LabelBinarizer()
    y = df[:, class_col]

    y_int = encoder.fit_transform(y)
    y_label_lookup = dict()
    y_label_lookup_inverse = dict()
    for index, l in zip(y_int.argmax(1), y):
        y_label_lookup[index] = l
        y_label_lookup_inverse[l] = index

    return y_int,y_label_lookup_inverse


def data_generator_mtl(df, class_col: int,classes: dict,
                       aux_class_cols:list, aux_classes:list,
                       batch_size, text_norm_option, embedding_model,
                   text_input_info: dict, embedding_format, shuffle=True, word_weights:list=None):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0  # Counter inside the current batch vector
    batch_x_multinput = []
    for k in text_input_info.keys():
        batch_x_multinput.append([])

    batch_y = None
    batch_ys_aux=[]

    while True:  # Loop forever
        if shuffle:
            np.random.shuffle(df)  # Shuffle df each epoch

        for i in range(len(df)):
            row = df[i]
            if batch_y is None:
                #init y labels for this batch
                batch_y = np.zeros((batch_size, len(classes)), dtype='int')
                for aux_class in aux_classes:
                    batch_y_aux=np.zeros((batch_size, len(aux_class)), dtype='int')
                    batch_ys_aux.append(batch_y_aux)

                #init x data for this batch
                for b in range(
                        len(text_input_info)):  # initiate each channel of input as a text_length x text_dim matrix
                    info = text_input_info[b]
                    batch_x_multinput[b] = np.zeros((batch_size, info["text_length"], info["text_dim"]),
                                                    dtype='float32')

            #go through each input channel (e.g., name, desc, cat)
            for b in range(len(text_input_info)):
                info = text_input_info[b]

                if embedding_format == 'fasttext':
                    batch_x_multinput[b][batch_i] = cl_dnn.text_to_vector_fasttext(cl_dnn.concate_text(row, info["text_col"]),
                                                                            embedding_model,
                                                                            info["text_length"], info["text_dim"],
                                                                            text_norm_option,word_weights)
                else:
                    batch_x_multinput[b][batch_i] = cl_dnn.text_to_vector_gensim(cl_dnn.concate_text(row, info["text_col"]),
                                                                          embedding_model,
                                                                          info["text_length"], info["text_dim"],
                                                                          text_norm_option,word_weights)
            # create the label vector
            cls = row[class_col]
            cls_index = classes[cls]
            batch_y[batch_i][cls_index] = 1
            # do so for the aux tasks
            for a in range(len(aux_class_cols)):
                aux_class = aux_classes[a]
                aux_class_col = aux_class_cols[a]
                ac = row[aux_class_col]
                ac_index = aux_class[ac]
                batch_ys_aux[a][batch_i][ac_index]=1


            batch_i += 1
            if batch_i == batch_size:
                # Ready to yield the batch
                batch_x_out=[batch_x_multinput]
                batch_y_out=[batch_y]
                for a in range(len(aux_class_cols)):
                    batch_x_aux= copy.deepcopy(batch_x_multinput)
                    batch_y_aux=batch_ys_aux[a]
                    batch_x_out.append(batch_x_aux)
                    batch_y_out.append(batch_y_aux)
                yield batch_x_multinput, batch_y_out

                batch_x_multinput = []
                for k in text_input_info.keys():
                    batch_x_multinput.append([])
                batch_y = None
                batch_ys_aux = []
                batch_i = 0


def fit_dnn_mtl(df: DataFrame, nfold: int, main_class_col: int,
                aux_class_cols:list,
                final_model: Model, outfolder: str,
                task: str, model_descriptor: str,
                text_norm_option: int, text_input_info: dict, embedding_model, embedding_model_format,
                word_weights:list=None):
    y_int,y_label_lookup_inverse=encode_labels(df, main_class_col)

    aux_y_int_list=[]
    aux_label_lookup_list=[]
    for c in aux_class_cols:
        aux_y_int, aux_y_label_lookup_inverse = encode_labels(df, c)
        aux_y_int_list.append(aux_y_int)
        aux_label_lookup_list.append(aux_y_label_lookup_inverse)

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
        training_generator = data_generator_mtl(df=X_train_merge_,
                                                class_col=main_class_col,
                                                classes=y_label_lookup_inverse,
                                                aux_class_cols=aux_class_cols,
                                                aux_classes=aux_label_lookup_list,
                                                batch_size=dmc.DNN_BATCH_SIZE,
                                                text_norm_option=text_norm_option,
                                                embedding_model=embedding_model,
                                                text_input_info=text_input_info,
                                                embedding_format=embedding_model_format,
                                                word_weights=word_weights)

        training_steps_per_epoch = round(len(X_train_merge_) / dmc.DNN_BATCH_SIZE)

        nfold_model.fit_generator(training_generator, steps_per_epoch=training_steps_per_epoch,
                                  epochs=dmc.DNN_EPOCHES)

        test_generator = data_generator_mtl(df=X_test_merge_,
                                            class_col=main_class_col,
                                            classes=y_label_lookup_inverse,
                                            aux_class_cols=aux_class_cols,
                                            aux_classes=aux_label_lookup_list,
                                            batch_size=len(X_test_merge_),
                                            text_norm_option=text_norm_option,
                                            embedding_model=embedding_model,
                                            text_input_info=text_input_info,
                                            embedding_format=embedding_model_format, shuffle=False,
                                            word_weights=word_weights)
        prediction_prob = nfold_model.predict_generator(test_generator, steps=1)

        # evaluate the model, just focus on the main task
        #
        predictions = prediction_prob[0][0].argmax(axis=-1)

        for i, l in zip(X_test_index, predictions):
            nfold_predictions[i] = l

        del nfold_model

    indexes = sorted(list(nfold_predictions.keys()))
    predicted_labels = []
    for i in indexes:
        predicted_labels.append(nfold_predictions[i])
    util.save_scores(predicted_labels, y_int.argmax(1), "dnn", task, model_descriptor, 3,
                     outfolder)