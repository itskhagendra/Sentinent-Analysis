import re
import pickle
import pandas as pd

from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


data = pd.read_csv('../Dataset/tweets.csv', encoding='cp1252')
data = data[['SentimentText', 'Sentiment']]
data['Sentiment'] = data['Sentiment'].map({0: "Negative", 1: "Positive"})
data.columns = ['text', 'sentiment']
print(data.head)

data['text'] = data['text'].apply(lambda z: z.lower())
data['text'] = data['text'].apply(lambda z: re.sub('[^a-zA-Z0-9\s]', '', z))

print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == 'Negative'].size)
