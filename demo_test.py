import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

tf.logging.set_verbosity(tf.logging.ERROR)

print("Loading Ai Engine ....")
model = load_model("./Models/big_dataset_model.h5")
print("Engine Loaded Successfully")

print("Loading Support Dependencies ....")
with open('./Models/big_dataset_token.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
print("Dependencies loaded successfully")

txt = input("Enter Your Thoughts ")
twt = [txt]

twt = tokenizer.texts_to_sequences(twt)
twt = pad_sequences(twt, maxlen=40, dtype='int32', value=0)
sentiment = model.predict(twt, batch_size=1, verbose=0)[0]
print(sentiment)

if np.argmax(sentiment) == 0:
    print("Negative")
elif np.argmax(sentiment) == 1:
    print("Positive")
