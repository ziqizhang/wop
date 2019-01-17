import functools
import random

from keras.layers import Embedding, concatenate
import numpy
from keras import Model, Sequential
from keras.layers import Concatenate, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, \
    Dense, Flatten, K
from keras.regularizers import L1L2

from sklearn.feature_extraction.text import CountVectorizer
import pickle
from feature import nlp
from data import data_util as du

'''
model_descriptor is parsed by 'parse_model_descriptor' method to create a Keras model object. some examples of the descriptors below

cnn examples: 
1) cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=2-softmax
2) cnn[2,3,4](conv1d=100,dropout=0.2)|maxpooling1d=4|flatten|dense=2-softmax

#2 means creating 3 parallel conv1d layer with 100 filters and window size of 2,3,4 respectively, and each is attached a 
dropout layer with dropout ratio of 0.2

scnn examples:
1) scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=2-softmax
2) scnn[2,3,4](conv1d=100,dropout=0.2)|maxpooling1d=4|flatten|dense=2-softmax

lstm examples:

- embedding layer is the default first layer and do not need to be included in the descriptor
- currently the model descriptor parser only accepts cnn as the immediate layer after embedding. The format of the cnn or
  scnn descriptor is cnn/scnn[window_size_seprated_by_comma](convlayername=filtersize,other_layers_after_conv_separated_by_comma)
- other layers are then stacked over cnn and separated by the character |


'''

DNN_EMBEDDING_DIM=300
DNN_MAX_SEQUENCE_LENGTH=100
DNN_EPOCHES=40
DNN_BATCH_SIZE=50
#DNN_MODEL_DESCRIPTOR= "cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="lstm=100-False|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="bilstm=100-False|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv"
#DNN_MODEL_DESCRIPTOR="scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"

def create_model(model_descriptor: str, max_index, word_embedding_dim=DNN_EMBEDDING_DIM,
                 max_sequence_length=DNN_MAX_SEQUENCE_LENGTH, wemb_matrix=None, append_feature_matrix=None):
    '''no pre-trained embedding is provided'''
    if wemb_matrix is None:
        if append_feature_matrix is not None: #if we want to concat word embedding with any other features
            embedding_layers = [Embedding(input_dim=max_index, output_dim=word_embedding_dim,
                                          input_length=max_sequence_length),
                                Embedding(input_dim=max_index, output_dim=len(append_feature_matrix[0]),
                                          weights=[append_feature_matrix],
                                          input_length=max_sequence_length,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=word_embedding_dim,
                                          input_length=max_sequence_length)]

    else:
        if append_feature_matrix is not None: #using pre-trained word embeddings
            concat_matrices = concatenate_matrices(wemb_matrix, append_feature_matrix)
            # load pre-trained word embeddings into an Embedding layer
            # note that we set trainable = False so as to keep the embeddings fixed
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(concat_matrices[0]),
                                          weights=[concat_matrices],
                                          input_length=max_sequence_length,
                                          trainable=False)]
        else:
            embedding_layers = [Embedding(input_dim=max_index, output_dim=len(wemb_matrix[0]),
                                          weights=[wemb_matrix],
                                          input_length=max_sequence_length,
                                          trainable=False)]


    model = parse_model_descriptor(embedding_layers, model_descriptor)

    return model


def parse_model_descriptor(embedding_layers, model_descriptor:str):
    "sub_conv[2,3,4](dropout=0.2,conv1d=100-v,)"
    layer_descriptors = model_descriptor.split("|")

    first_layer=layer_descriptors[0]

    '''example: cnn[2,3,4](conv1d=100,dropout=0.2)'''
    if first_layer.startswith("cnn") or first_layer.startswith("scnn"):
        window_size_str=first_layer[first_layer.index("[")+1: first_layer.index("]")]
        cnn_layer_desc = first_layer[first_layer.index("(")+1:len(first_layer)-1]
        cnns = []
        for w in window_size_str.split(","):
            cnn_components = cnn_layer_desc.split(",")
            if not cnn_components[0].startswith("conv1d"):
                raise ValueError('Error in the model descriptor for cnn layers. It must follow the pattern `conv1d=[filter]`: %s'%cnn_components[0])
            cnn_components[0] = cnn_components[0]+"-"+w

            if first_layer.startswith("cnn"):
                cnns.append(create_sequential_model(cnn_components,
                                                embedding_layers=embedding_layers))
            else:
                for mod in create_skipped_cnn_submodels(cnn_components, embedding_layers, int(w)):
                    cnns.append(mod)

    else: #assuming a sequential model
        model= create_sequential_model(layer_descriptors, embedding_layers=embedding_layers)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    cnns_outputs = [model.output for model in cnns]
    if len(cnns_outputs)>1:
        x = Concatenate(axis=1)(cnns_outputs)
    else:
        x=cnns_outputs[0]

    parallel_layers=Model(inputs=embedding_layers[0].input, outputs=x)
    #print("submodel:")
    #parallel_layers.summary()
    #print("\n")

    layer_descriptors.pop(0)
    big_model = Sequential()
    big_model.add(parallel_layers)
    create_sequential_model(layer_descriptors, big_model)

    big_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    big_model.summary()

    return big_model


def create_sequential_model(layer_descriptors:list(), model:Sequential=None,embedding_layers=None,  cnn_dilation=None):
    if model is None:
        model = Sequential()


    if embedding_layers is not None:
        if len(embedding_layers)==1:
            model.add(embedding_layers[0])
        else:
            concat_embedding_layers(embedding_layers, model)

    for layer_descriptor in layer_descriptors:
        ld=layer_descriptor.split("=")

        layer_name=ld[0]
        params=None
        if len(ld)>1:
            params=ld[1].split("-")

        if layer_name=="dropout":
            model.add(Dropout(float(params[0])))
        elif layer_name=="lstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(LSTM(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="gru":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(GRU(units=int(params[0]), return_sequences=return_seq))
        elif layer_name=="bilstm":
            if params[1]=="True":
                return_seq=True
            else:
                return_seq=False
            model.add(Bidirectional(LSTM(units=int(params[0]), return_sequences=return_seq)))
        elif layer_name=="conv1d":
            if cnn_dilation is None:
                model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(params[1]), padding='same', activation='relu'))
            else:
                model.add(Conv1D(filters=int(params[0]),
                             kernel_size=int(params[1]), dilation_rate=int(cnn_dilation),
                                 padding='same', activation='relu'))
        elif layer_name=="maxpooling1d":
            size=int(params[0])
            model.add(MaxPooling1D(pool_size=size))
        elif layer_name=="gmaxpooling1d":
            model.add(GlobalMaxPooling1D())
        elif layer_name == "dense":
            if len(params) == 2:
                model.add(Dense(int(params[0]), activation=params[1]))
            elif len(params) > 2:
                kernel_reg = create_regularizer(params[2])
                activity_reg = create_regularizer(params[3])
                if kernel_reg is not None and activity_reg is None:
                    model.add(Dense(int(params[0]), activation=params[1],
                                        kernel_regularizer=kernel_reg))
                elif activity_reg is not None and kernel_reg is None:
                    model.add(Dense(int(params[0]), activation=params[1],
                                        activity_regularizer=activity_reg))
                elif activity_reg is not None and kernel_reg is not None:
                    model.add(Dense(int(params[0]), activation=params[1],
                                        activity_regularizer=activity_reg,
                                        kernel_regularizer=kernel_reg))
        elif layer_name == "flatten":
            model.add(Flatten())
    return model

def create_skipped_cnn_submodels(layer_descriptors, embedding_layers, cnn_ks):
    if cnn_ks > 5:
        raise ValueError('Skip cnn window of >5 is not supported.')
    models = []
    conv_layers = []

    conv1d_desc = layer_descriptors[0]
    filter = int(conv1d_desc.split("=")[1].split("-")[0])
    layer_descriptors.pop(0)

    create_skipped_cnn_layers(cnn_ks,filter,conv_layers)

    for conv_layer in conv_layers:
        model = Sequential()
        if len(embedding_layers)==1:
            model.add(embedding_layers[0])
        else:
            concat_embedding_layers(embedding_layers, model)
        model.add(Dropout(0.2)) #try removing this
        model.add(conv_layer)
        create_sequential_model(layer_descriptors,model)
        models.append(model)

    return models

def create_skipped_cnn_layers(cnn_ks, filter:int, conv_layers:list, layer=None):
    if cnn_ks < 3:
        if layer is None:
            conv1d_3 = Conv1D(filters=filter, kernel_size=cnn_ks, padding='same', activation='relu')
        else:
            conv1d_3 = Conv1D(filters=filter, kernel_size=cnn_ks, padding='same', activation='relu')(layer)
        conv_layers.append(conv1d_3)
    elif cnn_ks == 3:
        if layer is None:
            conv1d_3 = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')
        else:
            conv1d_3 = Conv1D(filters=filter, kernel_size=3, padding='same', activation='relu')(layer)
        conv_layers.append(conv1d_3)

        # 2skip1
        ks_and_masks = generate_ks_and_masks(2, 1)
        for mask in ks_and_masks[1]:
            if layer is None:
                conv_layers.append(SkipConv1D(filters=filter,
                                          kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                          padding='same', activation='relu'))
            else:
                conv_layers.append(SkipConv1D(filters=filter,
                                              kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                              padding='same', activation='relu')(layer))

    elif cnn_ks == 4:
        if layer is None:
            conv1d_4 = Conv1D(filters=filter, kernel_size=4, padding='same', activation='relu')
        else:
            conv1d_4 = Conv1D(filters=filter, kernel_size=4, padding='same', activation='relu')(layer)
        conv_layers.append(conv1d_4)

        # 2skip2
        ks_and_masks = generate_ks_and_masks(2, 2)
        for mask in ks_and_masks[1]:
            if layer is None:
                conv_layers.append(SkipConv1D(filters=filter,
                                          kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                          padding='same', activation='relu'))
            else:
                conv_layers.append(SkipConv1D(filters=filter,
                                              kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                              padding='same', activation='relu')(layer))
        # 3skip1
        ks_and_masks = generate_ks_and_masks(3, 1)
        for mask in ks_and_masks[1]:
            if layer is None:
                conv_layers.append(SkipConv1D(filters=filter,
                                          kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                          padding='same', activation='relu'))
            else:
                conv_layers.append(SkipConv1D(filters=filter,
                                              kernel_size=int(ks_and_masks[0]), validGrams=mask,
                                              padding='same', activation='relu')(layer))



def generate_ks_and_masks(target_cnn_ks, skip):
    masks=[]
    real_cnn_ks=target_cnn_ks+skip
    for gap_index in range(1, real_cnn_ks):
        mask=[]
        for ones in range(0,gap_index):
            mask.append(1)
        for zeros in range(gap_index,gap_index+skip):
            if zeros<real_cnn_ks:
                mask.append(0)
        for ones in range(gap_index+skip, real_cnn_ks):
            if ones <real_cnn_ks:
                mask.append(1)

        if mask[len(mask)-1]!=0:
            masks.append(mask)
    return [real_cnn_ks,masks]

def concat_embedding_layers(embedding_layers, big_model):
    submodels = []

    for el in embedding_layers:
        m = Sequential()
        m.add(el)
        submodels.append(m)

    submodel_outputs = [model.output for model in submodels]
    if len(submodel_outputs) > 1:
        x = Concatenate(axis=2)(submodel_outputs)
    else:
        x = submodel_outputs[0]

    parallel_layers = Model(inputs=[embedding_layers[0].input, embedding_layers[1].input], outputs=x)
    big_model.add(parallel_layers)


def create_regularizer(string):
    if string=="none":
        return None
    string_array=string.split("_")
    return L1L2(float(string_array[0]),float(string_array[1]))


def build_pretrained_embedding_matrix(word_vocab: dict, model, expected_emb_dim, randomize_strategy
                                      ):
    # logger.info("\tloading pre-trained embedding model... {}".format(datetime.datetime.now()))
    # logger.info("\tloading complete. {}".format(datetime.datetime.now()))

    randomized_vectors = {}
    random_candidates=[] #list of word indexes in the embedding model to be randomly chosen
    words_matched=set() #track words that already matched and whose vectors are already used

    matrix = numpy.zeros((len(word_vocab), expected_emb_dim))
    count = 0
    randomized = 0
    for word, i in word_vocab.items():
        is_in_model = False
        if word in model.wv.vocab.keys():
            is_in_model = True
            vec = model.wv[word]
            matrix[i] = vec
            words_matched.add(word)

        if not is_in_model:
            randomized += 1
            if randomize_strategy == '1' or randomize_strategy == 1:  # randomly set values following a continuous uniform distribution
                vec = numpy.random.random_sample(expected_emb_dim)
                matrix[i] = vec
            elif randomize_strategy == '2' or randomize_strategy == 2:  # randomly take a vector from the model
                if word in randomized_vectors.keys():
                    vec = randomized_vectors[word]
                else:
                    if len(random_candidates)==0:
                        for x in range(0, len(model.wv.vocab.keys())):
                            random_candidates.append(x)
                        random.Random(4).shuffle(random_candidates)

                    while (True):
                        index = random_candidates.pop()
                        word = model.index2word[index]
                        if not word in words_matched:
                            words_matched.add(word)
                            break

                    vec = model.wv[word]
                    randomized_vectors[word] = vec

                matrix[i] = vec
        count += 1
        if count % 100 == 0:
            print(count)
    if randomize_strategy != '0':
        print("randomized={}".format(randomized))
    else:
        print("oov={}".format(randomized))

    return matrix

def concatenate_matrices(matrix1, matrix2):
    concat = numpy.concatenate((matrix1,matrix2), axis=1)
    return concat

def get_word_vocab(tweets:list, normalize_option, use_saved_vocab=False,
                   tweets_extra=None):
    word_vectorizer = CountVectorizer(
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=normalize_option),
        preprocessor=nlp.normalize_tweet,
        ngram_range=(1, 1),
        stop_words=nlp.stopwords,  # We do better when we keep stopwords
        decode_error='replace',
        max_features=500000,
        min_df=1,
        max_df=0.99
    )

    training_data_instances=len(tweets)
    if tweets_extra is not None:
        tweets.extend(tweets_extra)

    tweets=du.replace_nan_in_list(tweets)

    # logger.info("\tgenerating word vectors, {}".format(datetime.datetime.now()))
    counts = word_vectorizer.fit_transform(tweets).toarray()
    # logger.info("\t\t complete, dim={}, {}".format(counts.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
    word_embedding_input = []

    if not use_saved_vocab:
        with open('vocab.pickle', 'wb') as handle:
            pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for x in range(0, training_data_instances):
            tweet = counts[x]
            tweet_vocab = []
            for i in range(0, len(tweet)):
                if tweet[i] != 0:
                    tweet_vocab.append(i)
            word_embedding_input.append(tweet_vocab)
    else:
        with open('vocab.pickle', 'rb') as handle:
            saved_vocab = pickle.load(handle)
            vocab_reversed={val:key for (key, val) in vocab.items()}

            count_tweets=0
            for tweet in counts:
                count_tweets+=1
                if count_tweets%200==0:
                    print(count_tweets)
                tweet_vocab = []

                for i in range(0, len(tweet)):
                    if tweet[i] != 0:
                        vocab_index =i
                        v=vocab_reversed[vocab_index]
                        if v in saved_vocab.keys():
                            new_index=saved_vocab[v]
                            tweet_vocab.append(new_index)

                word_embedding_input.append(tweet_vocab)

    return word_embedding_input, vocab

def create_submodel_metafeature(inputs, dim):
    #flat1 = Flatten()(input_features)
    hidden = Dense(dim)(inputs)
    #model = Model(inputs=input_features, outputs=hidden)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return inputs


#model_option:
# 0= "cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"
# 1="lstm=100-False|dense=6-softmax|glv"
# 2="bilstm=100-False|dense=6-softmax|glv"
# 3="scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten|dense=6-softmax|glv"
# 4="scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten|dense=6-softmax|glv"

def create_submodel_textfeature(
                                inputs,
                                vocab_size, embedding_dim, max_seq_length, weights,
                                model_option):

    embedding = Embedding(vocab_size, embedding_dim, input_length=max_seq_length,
                          weights=[weights], trainable=False)(inputs)
    if model_option.startswith("cnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten"):
        #conv1d_1 = Conv1D(filters=100,
        #                  kernel_size=1, padding='same', activation='relu')(embedding)
        conv1d_2=Conv1D(filters=100,
               kernel_size=2, padding='same', activation='relu')(embedding)
        conv1d_3 = Conv1D(filters=100,
                          kernel_size=3, padding='same', activation='relu')(embedding)
        conv1d_4 = Conv1D(filters=100,
                          kernel_size=4, padding='same', activation='relu')(embedding)
        # conv1d_5 = Conv1D(filters=100,
        #                   kernel_size=5, padding='same', activation='relu')(embedding)
        # conv1d_6 = Conv1D(filters=100,
        #                   kernel_size=6, padding='same', activation='relu')(embedding)
        merge = concatenate([conv1d_2, conv1d_3,conv1d_4])
        pool=MaxPooling1D(pool_size=4)(merge)
        flat=Flatten()(pool)
        #final = Dense(targets, activation="softmax")(pool)
        #model = Model(inputs=deep_inputs, outputs=final)
        return flat
    elif model_option.startswith("lstm=100-False"):
        lstm=LSTM(units=100, return_sequences=False)(embedding)
        #final = Dense(targets, activation="softmax")(lstm)
        #model = Model(inputs=deep_inputs, outputs=final)
        #flat = Flatten()(lstm)
        return lstm
    elif model_option.startswith("bilstm=100-False"):
        lstm=Bidirectional(LSTM(units=100, return_sequences=False))(embedding)
        #final = Dense(targets, activation="softmax")(lstm)
        #model = Model(inputs=deep_inputs, outputs=final)
        #flat = Flatten()(lstm)
        return lstm
    elif model_option.startswith("scnn[2,3,4](conv1d=100,maxpooling1d=4)|maxpooling1d=4|flatten"):
        start=model_option.index("[")+1
        end=model_option.index("]")
        window_str = model_option[start:end]
        dropout=Dropout(0.2)(embedding)
        conv_layers_with_pooling=[]
        for i in window_str.split(","):
            ws=int(i)
            conv_layers=[]
            create_skipped_cnn_layers(ws, 100, conv_layers, dropout)
            for conv in conv_layers:
                conv_layers_with_pooling.append(MaxPooling1D(4)(conv))
        merge=concatenate(conv_layers_with_pooling)
        pool=MaxPooling1D(pool_size=4)(merge)
        #final = Dense(targets, activation="softmax")(pool)
        #model = Model(inputs=deep_inputs, outputs=final)
        flat = Flatten()(pool)
        return flat
    elif model_option.startswith("scnn[2,3,4](conv1d=100)|maxpooling1d=4|flatten"):
        start=model_option.index("[")+1
        end=model_option.index("]")
        window_str = model_option[start:end]
        dropout=Dropout(0.2)(embedding)
        conv_layers = []
        for i in window_str.split(","):
            ws=int(i)
            create_skipped_cnn_layers(ws, 100, conv_layers, dropout)
        merge=concatenate(conv_layers)
        pool=MaxPooling1D(pool_size=4)(merge)
        #final = Dense(targets, activation="softmax")(pool)
        #model = Model(inputs=deep_inputs, outputs=final)
        flat = Flatten()(pool)
        return flat
    else:
        raise ValueError("model option not supported: %s"%model_option)


#a 1D convolution that skips some entries
class SkipConv1D(Conv1D):

    #in the init, let's just add a parameter to tell which grams to skip
    def __init__(self, validGrams, **kwargs):

        #for this example, I'm assuming validGrams is a list
        #it should contain zeros and ones, where 0's go on the skip positions
        #example: [1,1,0,1] will skip the third gram in the window of 4 grams
        assert len(validGrams) == kwargs.get('kernel_size')
        self.validGrams = K.reshape(K.constant(validGrams),(len(validGrams),1,1))
            #the chosen shape matches the dimensions of the kernel
            #the first dimension is the kernel size, the others are input and ouptut channels


        #initialize the regular conv layer:
        super(SkipConv1D,self).__init__(**kwargs)

        #here, the filters, size, etc, go inside kwargs, so you should use them named
        #but you may make them explicit in this __init__ definition
        #if you think it's more comfortable to use it like this


    #in the build method, let's replace the original kernel:
    def build(self, input_shape):

        #build as the original layer:
        super(SkipConv1D,self).build(input_shape)

        #replace the kernel
        self.originalKernel = self.kernel
        self.kernel = self.validGrams * self.originalKernel

