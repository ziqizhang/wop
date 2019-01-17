import numpy
import pandas as pd
from feature import text_feature_extractor as tfe

# this is a text based feature loader
def create_textprofile(csv_basic_feature):
    text_col = 22  # 16-names; 22-profiles
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    df.astype(str)

    texts = df[:, text_col]
    texts = ["" if type(x) is float else x for x in texts]
    # Convert feature vectors to float64 type
    X_ngram, vocab = tfe.get_ngram_tfidf(texts)

    return X_ngram, y


def create_textprofileandname(csv_basic_feature):
    profile_col = 22  # 16-names; 22-profiles
    name_col = 15
    label_col = 40

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    y = df[:, label_col]

    df.astype(str)

    profiles = df[:, profile_col]
    profiles = ["" if type(x) is float else x for x in profiles]

    names = df[:, name_col]
    names = ["" if type(x) is float else x for x in names]
    names = [str(i) for i in names]
    # Convert feature vectors to float64 type
    profile_ngram, vocab = tfe.get_ngram_tfidf(profiles)
    name_ngram, vocab = tfe.get_ngram_tfidf(names)

    X_ngram = numpy.concatenate([name_ngram, profile_ngram], axis=1)

    return X_ngram, y

# this is the basic feature loader, using only the stats from indexed data.
def create_basic(csv_basic_feature, contains_label=True):
    feature_start_col = 1
    feature_end_col = 13

    y=None
    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0, quotechar='"').as_matrix()
    if contains_label:
        label_col = 40
        y = df[:, label_col]


    X = df[:, feature_start_col:feature_end_col + 1]
    # Convert feature vectors to float64 type
    X = X.astype(numpy.float32)

    return X, y


'''basic numeric features and tweet based numeric features (disease in tweets, topical tweets)'''
def create_numeric(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)
    csv_diseases_in_tweets = folder_other + \
                                   "/tweet_feature/diseases_in_tweets.csv"

    df = pd.read_csv(csv_diseases_in_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 2:]
    X_2 = X_2.astype(numpy.float32)
    numpy.nan_to_num(X_2,False)
    X_new = numpy.concatenate([X, X_2], axis=1)  # you can keep concatenating other matrices here. but
    # remember to call 'astype' like above on them before concatenating

    csv_topical_tweets = folder_other + \
                         "/tweet_feature/topical_tweets.csv"
    df = pd.read_csv(csv_topical_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 2:]
    X_2 = X_2.astype(numpy.float32)
    numpy.nan_to_num(X_2, False)
    X_new = numpy.concatenate([X_new, X_2], axis=1)

    return X_new, y


def create_manual_dict(csv_basic_feature, folder_other):

    X, y = create_basic(csv_basic_feature)

    csv_manual_dict_feature = folder_other + \
                              "/manual_dict_feature_1/feature_manualcreated_dict_match_profile.csv"
    df = pd.read_csv(csv_manual_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    return X_2, y


def create_autocreated_dict(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)

    csv_autocreated_dict_feature = folder_other + \
                                   "/dictionary_feature_1/feature_autocreated_dict_match_profile.csv"
    df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    return X_2, y


#auto dict, adds hashtags and words matched against profiles
def create_autocreated_dictext(csv_basic_feature, folder_other):
    X, y = create_basic(csv_basic_feature)

    csv_autocreated_dict_feature = folder_other + \
                                   "/dictionary_feature_1/feature_autocreated_dict_match_profile.csv"
    df = pd.read_csv(csv_autocreated_dict_feature, header=0, delimiter=",", quoting=0).as_matrix()
    X_2 = df[:, 1:]
    X_2 = X_2.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_2,False)

    csv_tweet_hashtag = folder_other + \
                        "/dictionary_feature_1/feature_disease_hashtag_match_profile.csv"
    df = pd.read_csv(csv_tweet_hashtag, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_hashtag = df[:, 1:]
    X_tweet_hashtag = X_tweet_hashtag.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_hashtag, False)

    csv_tweet_word = folder_other + \
                     "/dictionary_feature_1/feature_disease_word_match_profile.csv"
    df = pd.read_csv(csv_tweet_word, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_word = df[:, 1:]
    X_tweet_word = X_tweet_word.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_word, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_2, X_tweet_hashtag, X_tweet_word], axis=1)

    return X_all, y


def create_basic_and_user_url(csv_basic_feature):
    X, y = create_basic(csv_basic_feature)
    url_column = 23

    df = pd.read_csv(csv_basic_feature, header=0, delimiter=",", quoting=0).as_matrix()
    df = df.astype(str)
    X_2 = df[:, url_column]

    # change nan values to 0 and urls to 1
    X_2 = numpy.asarray([0 if x == "nan" else 1 for x in X_2])
    X_2 = X_2.reshape((len(X_2),1))
    X_2 = X_2.astype(numpy.float32)

    X_new = numpy.concatenate([X, X_2], axis=1)

    return X_new,y


def get_all_numeric_features(csv_basic_feature, folder_other):

    # get all numeric features

    X_basic, y = create_basic(csv_basic_feature)

    X_manual, _ = create_manual_dict(csv_basic_feature, folder_other)
    X_auto, _ = create_autocreated_dict(csv_basic_feature, folder_other)

    csv_diseases_in_tweets = folder_other + \
                             "/tweet_feature/diseases_in_tweets.csv"
    df = pd.read_csv(csv_diseases_in_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_diseases_tweets = df[:, 2:]
    X_diseases_tweets = X_diseases_tweets.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_diseases_tweets,False)


    csv_topical_tweets = folder_other + \
                         "/tweet_feature/topical_tweets.csv"
    df = pd.read_csv(csv_topical_tweets, header=0, delimiter=",", quoting=0).as_matrix()
    X_topical_tweets = df[:, 2:]
    X_topical_tweets = X_topical_tweets.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_topical_tweets,False)

    csv_manual_g = folder_other + \
                   "/manual_dict_feature_1/features_manual_dict_g.csv"
    df = pd.read_csv(csv_manual_g, header=0, delimiter=",", quoting=0).as_matrix()
    X_manual_g = df[:, 1:]
    X_manual_g = X_manual_g.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_manual_g,False)

    csv_tweet_hashtag = folder_other + \
                        "/dictionary_feature_1/feature_disease_hashtag_match_profile.csv"
    df = pd.read_csv(csv_tweet_hashtag, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_hashtag = df[:, 1:]
    X_tweet_hashtag = X_tweet_hashtag.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_hashtag,False)

    csv_tweet_word = folder_other + \
                     "/dictionary_feature_1/feature_disease_word_match_profile.csv"
    df = pd.read_csv(csv_tweet_word, header=0, delimiter=",", quoting=0).as_matrix()
    X_tweet_word = df[:, 1:]
    X_tweet_word = X_tweet_word.astype(numpy.float32)
    # replace nan values to 0
    numpy.nan_to_num(X_tweet_word,False)

    # concatenate all feature sets
    #basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_basic, X_manual, X_auto, X_diseases_tweets, X_topical_tweets, X_manual_g,
                               X_tweet_hashtag, X_tweet_word], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y


def create_text_and_numeric(csv_basic_feature, folder_other):
    X_numeric, y = create_numeric(csv_basic_feature,folder_other)
    X_text,_ = create_textprofile(csv_basic_feature)

    # replace nan values to 0
    numpy.nan_to_num(X_numeric, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_numeric], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y


def create_text_and_manualdict(csv_basic_feature, folder_other):
    X_manualdict, y = create_manual_dict(csv_basic_feature, folder_other)
    X_text, _ = create_textprofile(csv_basic_feature)

    # replace nan values to 0
    numpy.nan_to_num(X_manualdict, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_manualdict], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y


def create_text_and_autodict(csv_basic_feature, folder_other):
    X_autodict, y = create_autocreated_dict(csv_basic_feature, folder_other)
    X_text, _ = create_textprofile(csv_basic_feature)

    # replace nan values to 0
    numpy.nan_to_num(X_autodict, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_autodict], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y

def create_text_and_autodictext(csv_basic_feature, folder_other):
    X_autodictext, y = create_autocreated_dictext(csv_basic_feature, folder_other)
    X_text, _ = create_textprofile(csv_basic_feature)

    # replace nan values to 0
    numpy.nan_to_num(X_autodictext, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_autodictext], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y


def create_text_and_numeric_and_autodictext(csv_basic_feature, folder_other):
    X_numeric, y = create_numeric(csv_basic_feature, folder_other)
    X_autodictext, _ = create_autocreated_dictext(csv_basic_feature, folder_other)
    X_text, _ = create_textprofile(csv_basic_feature)

    # replace nan values to 0
    numpy.nan_to_num(X_autodictext, False)

    # concatenate all feature sets
    # basic + manual + auto + diseases + topical + manual_g + hashtag + word + generic1 + generic2
    X_all = numpy.concatenate([X_text, X_numeric,X_autodictext], axis=1)
    print("Size of the array: ")
    print(X_all.shape)

    return X_all, y