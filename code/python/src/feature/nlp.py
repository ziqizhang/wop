import re

import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

sentiment_analyzer = VS()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")
stopwords=stopwords+ ['http','tweet','retweet','rt','twitter','https','tweets']
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
text_processor = TextPreProcessor(
    # terms that will be normalized
    # normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
    #                'time', 'url', 'date', 'number'],
    #     # terms that will be annotated
    #     annotate={"hashtag", "allcaps", "elongated", "repeated",
    #               'emphasis', 'censored'},

    normalize=[],
    # terms that will be annotated
    annotate={'elongated',
              'emphasis'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def normalize_tweet(tweet_text):
    tweet_text = text_processor.pre_process_doc(tweet_text.strip())
    tweet_text = list(filter(lambda a: a != '<elongated>', tweet_text))
    tweet_text = list(filter(lambda a: a != '<emphasis>', tweet_text))
    tweet_text = list(filter(lambda a: a != 'RT', tweet_text))
    tweet_text = list(filter(lambda a: a != '"', tweet_text))
    tweet_text = " ".join(tweet_text)
    return tweet_text.strip()


# stem_or_lemma: 0 - apply porter's stemming; 1: apply lemmatization; 2: neither
# -set to 0 to reproduce Davidson. However, note that because a different stemmer is used, results could be
# sightly different
# -set to 2 will do 'basic_tokenize' as in Davidson
def tokenize(tweet, stem_or_lemma=0):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens=[]
    if stem_or_lemma==0:
        for t in tweet.split():
            if len(t)<4:
                tokens.append(t)
            else:
                tokens.append(stemmer.stem(t))
    elif stem_or_lemma==1:
        for t in tweet.split():
            if len(t)<4:
                tokens.append(t)
            else:
                tokens.append(lemmatizer.lemmatize(t))
    else:
        tokens = [str(t) for t in tweet.split()] #this is basic_tokenize in TD's original code
    return tokens


# tweets should have been preprocessed to the clean/right format before passing to this method
def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = tokenize(t, 2)
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags
