import datetime

import logging

from feature import text_feature_extractor as fe
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def text_to_ngram(*texts):
    """"
    This function takes a list of tweets, along with used to
    transform the tweets into the format accepted by the model.

    Each tweet is decomposed into
    (a) An array of TF-IDF scores for a set of n-grams in the tweet.
    (b) An array of POS tag sequences in the tweet.
    (c) An array of features including sentiment, vocab, and readability.

    Returns a pandas dataframe where each row is the set of features
    for a tweet. The features are a subset selected using a Logistic
    Regression with L1-regularization on the training data.

    """
    # Features group 1: tfidf weighted n-grams applied to different parts of text
    features_by_type = {}
    text_features = []
    count = 0
    for text_matrix in texts:
        td_tfidf = fe.get_ngram_tfidf(text_matrix)
        text_features.append(td_tfidf[0])
        count += 1
        features_by_type[count] = td_tfidf[0]
    M = np.concatenate(text_features, axis=1)

    logger.info("\t\tcompleted, {}, {}".format(M.shape, datetime.datetime.now()))

    # print(M.shape)
    return [pd.DataFrame(M), features_by_type]


def text_to_skipgram(*texts):
    logger.info("\tgenerating CHASE 1 skip bigram feature vectors, {}".format(datetime.datetime.now()))
    c_skipgram_21 = fe.get_skipgram(texts, 2, 1)

    logger.info("\tgenerating CHASE 2 skip bigram feature vectors, {}".format(datetime.datetime.now()))
    c_skipgram_22 = fe.get_skipgram(texts, 2, 2)

    logger.info("\tgenerating CHASE 2 skip trigram feature vectors, {}".format(datetime.datetime.now()))
    c_skipgram_31 = fe.get_skipgram(texts, 3, 1)
    logger.info("\tgenerating CHASE 2 skip trigram feature vectors, {}".format(datetime.datetime.now()))
    c_skipgram_32 = fe.get_skipgram(texts, 3, 2)

    features_by_type = {}
    features_by_type["1skip2gram"]=c_skipgram_21
    features_by_type["2skip2gram"] = c_skipgram_22
    features_by_type["1skip3gram"] = c_skipgram_31
    features_by_type["2skip3gram"] = c_skipgram_32
    M = np.concatenate([c_skipgram_21,
                        c_skipgram_22[0], c_skipgram_31[0],
                        c_skipgram_32], axis=1)

    return [pd.DataFrame(M), features_by_type]
