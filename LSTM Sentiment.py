# Required Modules
import re
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


# Data Pre-Processing
data = pd.read_csv('./Sentiment.csv')
data = data[['text', 'sentiment']]
data = data[data.sentiment != "Neutral"]

data['text'] = data['text'].apply(lambda z: z.lower())
data['text'] = data['text'].apply(lambda z: re.sub('[^a-zA-Z0-9\s]', '', z))

print(data[data['sentiment'] == 'Positive'].size)
print(data[data['sentiment'] == "Negative"].size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

# Tokenization
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data['text'].values)
x = tokenizer.texts_to_sequences(data['text'].values)
x = pad_sequences(x)

# Training Parameters
embed_dim = 128
lstm_out = 196
epoch = 50
batch_size = 32
validation_size = 1500


# Neural Network  Schema
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# Train And Test Set Generation
y = pd.get_dummies(data['sentiment']).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Training
model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)

# Validation Set Generation
x_validate = x_test[-validation_size:]
y_validate = y_test[-validation_size:]
x_test = x_test[:-validation_size]
y_test = y_test[:-validation_size]

# Model Validation
score, acc = model.evaluate(x_test, y_test, verbose=1, batch_size=batch_size)

print("Score:%.2f" % score)
print("Accuracy:%.2f" % acc)
