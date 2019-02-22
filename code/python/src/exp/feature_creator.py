import numpy
import pandas as pd
from feature import text_feature_extractor as tfe

# this is a text based feature loader
# MLP: I added a few more variables to support the new dataset
def create_features_text(training_text_features, text_col=22, label_col=40, data_delimiter=",", text_encoding="utf-8",
                         text_header=0):

    df = pd.read_csv(training_text_features, header=text_header, delimiter=data_delimiter, quoting=0,
                     encoding=text_encoding).as_matrix()
    y = df[:, label_col]

    df.astype(str)

    texts = df[:, text_col]
    texts = ["" if type(x) is float else x for x in texts]
    # this extracts ngram features from the profiles of the twitter user
    X_ngram, vocab = tfe.get_ngram_tfidf(texts)

    return X_ngram, y

# MLP: I added a new method to allow passing multiple column fields as the X data
def create_features_text_multiple_fields(training_text_features, text_cols, label_col, data_delimiter, text_encoding="utf-8",
                         text_header=0):

    df = pd.read_csv(training_text_features, header=text_header, delimiter=data_delimiter, quoting=0,
                     encoding=text_encoding).as_matrix()
    y = df[:, label_col]
    df.astype(str)

    texts = []

    # concatenate texts in the input fields (text_cols)
    for index in text_cols:
        if (len(texts) == 0):
            texts = df[:, index]
            texts = ["" if type(x) is float else x for x in texts]
        else:
            data = df[:, index]
            data = ["" if type(x) is float else x for x in data]
            texts = list(map('. '.join, zip(texts, data)))
    X_ngram, vocab = tfe.get_ngram_tfidf(texts)
    return X_ngram, y

def create_features_text_and_other(training_text_features, training_other_features):
    X_autodictext, y = create_features_gazetteer(training_text_features, training_other_features)
    X_text, _ = create_features_text(training_text_features)

    # replace nan values to 0
    numpy.nan_to_num(X_autodictext, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_autodictext], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y


#auto dict, adds hashtags and words matched against profiles
def create_features_gazetteer(training_text_features, training_other_features):
    X, y = create_features_text(training_text_features)

    folder_gazetteer_features = training_other_features + \
                                   "/gazetteer/dict1_match_to_profile.csv"
    df = pd.read_csv(folder_gazetteer_features, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    csv_tweet_hashtag = training_other_features + \
                        "/gazetteer/hashtag_match_to_profile.csv"
    df = pd.read_csv(csv_tweet_hashtag, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_hashtag = df[:, 1:]
    X_tweet_hashtag = X_tweet_hashtag.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_hashtag, False)


    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_2, X_tweet_hashtag], axis=1)

    return X_all, y


