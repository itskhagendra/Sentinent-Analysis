import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

data = pd.read_csv('./Sentiment.csv')
data = data[['text', 'sentiment']]
data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == "Negative"].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x)


embed_dim = 128
lstm_out = 196


# NN Schema
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

y = pd.get_dummies(data['sentiment']).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

batch_size = 32
model.fit(x_train, y_train, epochs=7, batch_size=batch_size, verbose=2)

validation_size = 1500
x_validate = x_test[-validation_size:]
y_validate = y_test[-validation_size:]
x_test = x_test[:-validation_size]
y_test = y_test[:-validation_size]

score, acc = model.evaluate(x_test, y_test, verbose=True, batch_size=batch_size)

print("Score:%.2f" % score)
print("Accuracy:%.2f" % acc)


