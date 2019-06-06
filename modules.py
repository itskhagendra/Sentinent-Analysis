# important Modules
import re
import gensim
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

LabelledSentense = gensim.models.doc2vec.LabeledSentence
tqdm.pandas(desc="Progress-bar")
tokenizer = TweetTokenizer()

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def ingest():
    print("Dataset Loading ...")
    data = pd.read_csv("./tweets.csv", encoding='cp1252')
    data.columns = ['sentiment', 'itemId', 'Date', 'query', 'sentimentSource', 'sentimentText']
    data.drop(['itemId', 'Date', 'query', 'sentimentSource'], axis=1, inplace=True)
    print("Dataset Loading Completed")
    return data


def remove_pattern(input_text, pattern):
    r = re.findall(pattern, input_text)
    for i in r:
        input_text = re.sub(i, '', input_text)
    return input_text


def cleaning(data):
    print("Data Cleaning Started...")
    df = pd.DataFrame(columns=['tidy'])
    df['tidy'] = np.vectorize(remove_pattern)(data['sentimentText'], "@[\w]*")
    df['tidy'] = df['tidy'].str.replace("[^a-zA-Z#]", " ")
    df['tidy'] = df['tidy'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
    print("Data Cleaning Finished")
    data = pd.concat([data, df], axis=1)
    data.reset_index()
    return data


def tokenize(tweet):
    print("Tokenization Started...")
    tokenized = tweet['tidy'].apply(lambda x: x.split())
    print("Tokenization Finished")

    print("Stremmer Process Started...")
    stemmer = PorterStemmer()
    tokenized = tokenized.apply(lambda x: [stemmer.stem(i) for i in x])
    print("Stemmer process Finished.")

    print("Token Stitching process started... ")
    for i in range(len(tokenized)):
        tokenized[i] = " ".join((tokenized[i]))
    print("Token Stitching Process Finished")

    return tokenized

def hashtag(df):
    hashtags = []
    for i in df:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def tf_idf(df):
    print("TF-IDF Vectorization Started...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
    tfidf_vectorizer = tfidf_vectorizer.fit_transform(df['tidy'])
    print("TF-IDF Vectorization Finished")
    return tfidf_vectorizer


def bow_features(df):
    print("Feature Extraction Started")
    bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(df['tidy'])
    print("Feature Extraction Finished")
    return bow




