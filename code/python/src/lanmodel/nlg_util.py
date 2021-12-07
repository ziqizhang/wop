# main source:https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275
# model source:https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb
import sys

import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku
import numpy as np
from classifier import classifier_util
tokenizer = Tokenizer()


def dataset_preparation(data):
    # basic cleanup
    corpus = data.lower().split("\n")
    print("\t corpus size={} sentences @{}".format(len(corpus), str(datetime.datetime.now())))
    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print("\t total words={} @{}".format(total_words, str(datetime.datetime.now())))

    # create input sequences using list of tokens
    input_sequences = []
    lines=0
    for line in corpus:
        lines+=1
        if lines%1000==0:
            print("\t\t{} lines".format(lines))
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    print("\t sequence input prepared @{}".format(str(datetime.datetime.now())))
    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # create predictors and label
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    print("\t training data prepared @{}".format(str(datetime.datetime.now())))

    return predictors, label, max_sequence_len, total_words


def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 300, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
    model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=20, batch_size=200, verbose=1, callbacks=[earlystop])
    model.summary()
    return model


def generate_text(model, seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

def slice_text_data(in_file, out_folder,parts):
    f = open(in_file)

    writers={}
    for i in range(parts):
        writers[i]=open(out_folder+"/"+str(i)+".txt",'w')

    line = f.readline()

    total_lines=0
    count=0
    while line:
        line = f.readline()

        wr = writers[count]
        wr.write(line)

        count+=1
        if count>=len(writers):
            count=0

        total_lines+=1
        if total_lines%100000==0:
            print(total_lines)

    f.close()

    for k, v in writers.items():
        v.close()


if __name__ == "__main__":

    slice_text_data("/home/zz/Work/data/wdc/desc_txt/desc_cleaned.txt","/home/zz/Work/data/wdc/desc_txt",100)
    exit(0)

    data = open(sys.argv[1]).read()

    predictors, label, max_sequence_len, total_words = dataset_preparation(data)
    model = create_model(predictors, label, max_sequence_len, total_words)
    classifier_util.save_classifier_model(model, sys.argv[2])


    text=generate_text(model, "we naughty", 100, max_sequence_len)
    print(text)