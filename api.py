
import flask
import json
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


app = flask.Flask(__name__)
model = None
tokenizer = None


def load_engine():
    global model, tokenizer
    with open('./Models/big_dataset_token.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model("./Models/big_dataset_model.h5")
    global graph
    graph = tf.get_default_graph()


def preprocess_text(text):
    user_input = [text]
    user_input = tokenizer.texts_to_sequences(user_input)
    user_input = pad_sequences(user_input, maxlen=40, dtype='int32', value=0)
    return user_input


def prediction(temp):
    final_sentiment = None
    user_input = preprocess_text(temp)
    with graph.as_default():
        sentiment = model.predict(user_input, batch_size=1, verbose=0)[0]

    if np.argmax(sentiment) == 0:
        final_sentiment = 'negative'
    elif np.argmax(sentiment) == 1:
        final_sentiment = 'positive'
    result = {
        "positive_confidence": str(sentiment[1]),
        "negative_confidence": str(sentiment[0]),
        'sentiment': str(final_sentiment)
    }
    return result

@app.route('/')
def home():
    return flask.render_template("index.html")


@app.route('/api', methods=['POST'])
def api():
    data = flask.request.form['text']
    result = prediction(data)
    return flask.jsonify(result)


if __name__ == '__main__':
    print("Please Wait We are making all systems ready")
    load_engine()
    app.run(host='127.0.0.1', port=8080)
