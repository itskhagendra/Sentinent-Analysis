# Sentiment-Analysis
 Given a message, classify whether the message is of positive, negative, or neutral
sentiment. For messages conveying both a positive and negative sentiment,
whichever is the stronger sentiment should be chosen.

## Motivation
- In the past decade, new forms of communication, such as microblogging and text messaging have emerged and
become ubiquitous. While there is no limit to the range of information conveyed by tweets and texts, often these
short messages are used to share opinions and sentiments that people have about what is going on in the world
around them.

- Tweets and texts are short: a sentence or a headline rather than a document. The language used is very informal,
with creative spelling and punctuation, misspellings, slang, new words, URLs, and genre-specific terminology and
abbreviations, such as, RT for `"re-tweet"` and `# hashtags`, which are a type of tagging for Twitter messages.

- Another aspect of social media data such as Twitter messages is that it includes rich structured information about
the individuals involved in the communication. For example, Twitter maintains information of who follows whom
and re-tweets and tags inside of tweets provide discourse information.

## Tools Used
- Keras
- Sklean
- Pickle

# How to use this 

1. Download the Model `Sentiment-negative_bias.h5` 
2. Download the Tokenizer file i.e `tokenizer.pickle`

3. Create a new file and write the following codes for importing dependencies

    ```
    import pickle
    import numpy as np
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences 
    ```

4. use the following code snippet for loading model and Tokenizer
    ```
    model = load_model("Sentiment-negative_bias.h5")
    print("Model Loaded Successfully")
    ```
    
    ```
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully")
    ```

5. Now convert your input string into list
6. Tokenize your input stream
    `twt = tokenizer.texts_to_sequences(twt)`
    
7. Add appropriate padding 
    `twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)`
8. predict the results 
    `sentiment = model.predict(twt, batch_size=1, verbose=0)[0]`
    
## Thank You 
if you liked it please considering adding a star
 