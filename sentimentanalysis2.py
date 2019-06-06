import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from numpy import array
from keras.models import load_model
import pickle
import re

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

# from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
# tokenizer = TweetTokenizer()

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

# can load model this way alternatively
loaded_model = load_model("model.h5")

# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(test_vecs_w2v, y_test, verbose=0)?
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print(loaded_model.summary())
print(loaded_model.get_weights())

def process(txt):
    out = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
    out = out.split()
    out = [word.lower() for word in out]
    return out

def tokenize(thresh = 5):
    count  = dict()
    idx = 1
    word_index = dict()
    for txt in data_text:
        words = process(txt)
        for word in words:
            if word in count.keys():
                count[word] += 1
            else:
                count[word]  = 1
    most_counts = [word for word in count.keys() if count[word]>=thresh]
    for word in most_counts:
        word_index[word] = idx
        idx+=1
    return word_index

data_text = ["Modi is best"]
a = tokenize()
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

query = tokenizer.texts_to_sequences(data_text)
query = pad_sequences(query, maxlen=50)

# input = np.array(["Modi is best"])
# print(query)
# print(type(query))
# print(np.shape(query))

# def labelizeTweets(tweets, label_type):
#     labelized = []
#     for i,v in tqdm(enumerate(tweets)):
#         label = '%s_%s'%(label_type,i)
#         labelized.append(LabeledSentence(v, [label]))
#     return labelized
#
# x_predict= labelizeTweets(query,'PREDICT')


# prediction
prediction = loaded_model.predict(x_predict)
print(prediction)
