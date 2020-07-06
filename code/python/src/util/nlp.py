import re
from wordsegment import load, segment

import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
sentiment_analyzer = VS()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")
stopwords=stopwords+ ['http','tweet','retweet','rt','twitter','https','tweets','home','sale','number']
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)
from urllib.parse import urlparse

def url_to_words(url:str):
    try:
        o = urlparse(url)
        host = o.hostname
        if host is None:
            host=""
        else:
            host=host.split(".")
            index=-1
            length=-1
            for i in range(len(host)):
                p = host[i]
                if len(p)>length:
                    length=len(p)
                    index=i
            if index>-1 and len(host[index])>3:
                host=host[index]
                host=" ".join(segment(host))

            else:
                host=""

        path= re.sub('[^0-9a-zA-Z]+', '*', o.path).strip()

        return host+" "+path

    except:
        return ""

#use regex to remove urls, and remove any non alphanumeric chars
def normalize(text:str):
    #if the text is entirely a url, use wordsegment
    if text.startswith("http") and " " not in text:
        text=url_to_words(text)

    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text=re.sub(r'\W+', ' ', text).strip()
    return text


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


if __name__ == "__main__":
    load()
    s=normalize("http://shop.georgiastatesports.com/Cold_Weather_Gear")
    print(s)

# text="http://schema.org/Product/offers\|http://schema.org/Product/name\|http://schema.org/Product/description\|http://schema.org/Product/url\|http://schema.org/Product/image NFL > Seattle Seahawks > Seattle Seahawks Flags & Banners"
# print(normalize(text))