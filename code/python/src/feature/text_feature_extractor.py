import datetime
import functools
import pickle

import logging
from nltk.util import skipgrams
from sklearn.feature_extraction.text import TfidfVectorizer

from feature import nlp

logger = logging.getLogger(__name__)
NGRAM_FEATURES_VOCAB="feature_vocab_ngram"
NGRAM_POS_FEATURES_VOCAB="feature_vocab_ngram_pos"
SKIPGRAM_FEATURES_VOCAB="feature_vocab_skipgram"
SKIPGRAM_POS_FEATURES_VOCAB="feature_vocab_skipgram_pos"
TWEET_TD_OTHER_FEATURES_VOCAB="feature_vocab_td_other"
TWEET_HASHTAG_FEATURES_VOCAB="feature_vocab_chase_hashtag"
TWEET_CHASE_STATS_FEATURES_VOCAB="feature_vocab_chase_stats"
SKIPGRAM22_FEATURES_VOCAB="feature_vocab_2skip_bigram"
SKIPGRAM32_FEATURES_VOCAB="feature_vocab_2skip_trigram"
SKIPGRAM22_POS_FEATURES_VOCAB="feature_vocab_pos_2skip_bigram"
SKIPGRAM32_POS_FEATURES_VOCAB="feature_vocab_pos_2skip_trigram"
SKIPGRAM_OTHERING="feature_skipgram_othering"

DNN_WORD_VOCAB="dnn_feature_word_vocab"

skipgram_label={}

ngram_vectorizer = TfidfVectorizer(
            # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            tokenizer=nlp.tokenize,
            preprocessor=nlp.normalize_tweet,
            ngram_range=(1, 4),
            stop_words=nlp.stopwords,  # We do better when we keep stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=20000,
            min_df=2,
            max_df=0.501
        )

#generates tfidf weighted ngram feature as a matrix and the vocabulary
def get_ngram_tfidf(texts):
    logger.info("\tgenerating n-gram vectors, {}".format(datetime.datetime.now()))
    tfidf = ngram_vectorizer.fit_transform(texts).toarray()
    logger.info("\t\t complete, dim={}, {}".format(tfidf.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(ngram_vectorizer.get_feature_names())}
    return tfidf, vocab

#tweets should be normalised already, as this method will be called many times to get different
#shape of skipgrams
def get_skipgram(tweets, nIn, kIn):
    #tokenization and preprocess (if not yet done) must be done here. when analyzer receives
    #a callable, it will not perform tokenization, see documentation
    tweet_tokenized=[]
    for t in tweets:
        tweet_tokenized.append(nlp.tokenize(t))
    skipper = functools.partial(skipgrams, n=nIn, k=kIn)
    vectorizer = TfidfVectorizer(
            analyzer=skipper,
            #stop_words=nlp.stopwords,  # We do better when we keep stopwords
            use_idf=True,
            smooth_idf=False,
            norm=None,  # Applies l2 norm smoothing
            decode_error='replace',
            max_features=10000,
            min_df=2,
            max_df=0.501
        )
    # for t in cleaned_tweets:
    #     tweetTokens = word_tokenize(t)
    #     skipgram_feature_matrix.append(list(skipper(tweetTokens)))

    # Fit the text into the vectorizer.
    logger.info("\tgenerating skip-gram vectors, n={}, k={}, {}".format(nIn, kIn,datetime.datetime.now()))
    tfidf = vectorizer.fit_transform(tweet_tokenized).toarray()
    logger.info("\t\t complete, dim={}, {}".format(tfidf.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    return tfidf, vocab
