import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re


class MaskClassifier:

    def __init__(self):
        self.model = self.load_model()
        self.token = self.load_tokens()
        self.sentiment = {
            0: 'positive',
            1: 'negative',
            2: 'neutral'
        }


    def load_model(self):
        model = load_model('covid_mask')
        return model


    def load_tokens(self):
        with open('token.json') as file:
            token_data = json.load(file)
            tokens = tokenizer_from_json(token_data)
        return tokens


    def preprocess(self, tweet):
        tweet = re.sub(r'http\S+', ' ', tweet)
        tweet = re.sub(r'#\w+', ' ', tweet)
        tweet = re.sub(r'@\w+', ' ', tweet)
        tweet = tweet.replace("'", '')
        tweet = tweet.lower()
        tweet = re.sub(r'[^a-zA-Z.?!]', ' ', tweet) 
        tweet = re.sub(r'\s{2,}', ' ', tweet)
        tweet = tweet.strip()
    
        return tweet


    def get_predictions(self, tweet: str):
        tweet = self.preprocess(tweet)
        tweet_sequence = self.token.texts_to_sequences([tweet])
        padded_tweet = pad_sequences(tweet_sequence, maxlen=75, padding='post', truncating='post')
        tweet_input = tf.expand_dims(padded_tweet[0], 0)
        output = self.model.predict(tweet_input)
        
        return {
            'sentiment': self.sentiment[np.argmax(output)],
            'probability': np.max(output)
        }