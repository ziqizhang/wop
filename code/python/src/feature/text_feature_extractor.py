import datetime
import functools

import logging
from nltk.util import skipgrams
from sklearn.feature_extraction.text import TfidfVectorizer

from util import nlp
import numpy as np

logger = logging.getLogger(__name__)
NGRAM_FEATURES_VOCAB = "feature_vocab_ngram"
NGRAM_POS_FEATURES_VOCAB = "feature_vocab_ngram_pos"
SKIPGRAM_FEATURES_VOCAB = "feature_vocab_skipgram"
SKIPGRAM_POS_FEATURES_VOCAB = "feature_vocab_skipgram_pos"
TWEET_TD_OTHER_FEATURES_VOCAB = "feature_vocab_td_other"
TWEET_HASHTAG_FEATURES_VOCAB = "feature_vocab_chase_hashtag"
TWEET_CHASE_STATS_FEATURES_VOCAB = "feature_vocab_chase_stats"
SKIPGRAM22_FEATURES_VOCAB = "feature_vocab_2skip_bigram"
SKIPGRAM32_FEATURES_VOCAB = "feature_vocab_2skip_trigram"
SKIPGRAM22_POS_FEATURES_VOCAB = "feature_vocab_pos_2skip_bigram"
SKIPGRAM32_POS_FEATURES_VOCAB = "feature_vocab_pos_2skip_trigram"
SKIPGRAM_OTHERING = "feature_skipgram_othering"

DNN_WORD_VOCAB = "dnn_feature_word_vocab"

skipgram_label = {}

ngram_vectorizer = TfidfVectorizer(
    # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    preprocessor=nlp.normalize,
    tokenizer=functools.partial(nlp.tokenize, stem_or_lemma=1),
    ngram_range=(1, 1),
    stop_words=nlp.stopwords,  # We do better when we keep stopwords
    use_idf=True,
    smooth_idf=False,
    norm=None,  # Applies l2 norm smoothing
    decode_error='replace',
    max_features=50000,
    min_df=2,
    max_df=0.99
)


# generates tfidf weighted ngram feature as a matrix and the vocabulary
def get_ngram_tfidf(texts):
    logger.info("\tgenerating n-gram vectors, {}".format(datetime.datetime.now()))
    tfidf = ngram_vectorizer.fit_transform(texts).toarray()
    logger.info("\t\t complete, dim={}, {}".format(tfidf.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(ngram_vectorizer.get_feature_names())}
    return tfidf, vocab

def concate_text(row: list, col_indexes):
    text = ""
    if "-" in col_indexes:
        for c in col_indexes.split("-"):
            text += row[int(c)] + " "
    else:
        text=row[int(col_indexes)]
    return text.strip()

# tweets should be normalised already, as this method will be called many times to get different
# shape of skipgrams
def get_skipgram(tweets, nIn, kIn):
    # tokenization and preprocess (if not yet done) must be done here. when analyzer receives
    # a callable, it will not perform tokenization, see documentation
    tweet_tokenized = []
    for t in tweets:
        tweet_tokenized.append(nlp.tokenize(t))
    skipper = functools.partial(skipgrams, n=nIn, k=kIn)
    vectorizer = TfidfVectorizer(
        analyzer=skipper,
        # stop_words=nlp.stopwords,  # We do better when we keep stopwords
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
    logger.info("\tgenerating skip-gram vectors, n={}, k={}, {}".format(nIn, kIn, datetime.datetime.now()))
    tfidf = vectorizer.fit_transform(tweet_tokenized).toarray()
    logger.info("\t\t complete, dim={}, {}".format(tfidf.shape, datetime.datetime.now()))
    vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    return tfidf, vocab


'''
aggr_option: 0 means average; 1 means sum
text_norm_option:0 means stemming 1 means lemma
'''
def get_aggr_embedding_vectors(df, text_col,
                               text_norm_option: int,
                               aggr_option: int, emb_model,
                               emb_format: str, emb_dim):
    X = np.zeros((len(df), emb_dim), dtype='float')
    i=0
    for row in df:
        text=concate_text(row, str(text_col))
        text = nlp.normalize(text)
        words = nlp.tokenize(text, text_norm_option)

        matrix=[]
        for w in words:
            # if w=='fidlar':
            #     print()
            if emb_format=='fasttext':
                vec=emb_model.get_word_vector(w).astype('float32')
            else:
                if w in emb_model.wv.vocab.keys():
                    vec = emb_model.wv[w]
                else:
                    vec=None
            if vec is not None:
                matrix.append(vec)

        if len(matrix)>0:
            if aggr_option==0:
                X[i, :]=np.sum(matrix, axis=0)
            else:
                X[i, :]=np.average(matrix, axis=0)

        #check if vec has nan
        # vec=X[i, :]
        # if np.isnan(vec).any():
        #     print("has NaN")
        i+=1
    return X