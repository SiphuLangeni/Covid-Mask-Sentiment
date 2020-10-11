import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model = load_model('mask_h5')


class MaskSentiment:
    def __init__(self, tweet: str):
        self.model: tf.keras.Models
        self.token = Tokenizer()
        self.tweet = tweet

    def preprocess(self):
        text = re.sub(r'@\w+', '', self.tweet)
        text = re.sub(r'http\S+', '', text)
        text = text.replace("'", '')
        text = text.lower()
        return text

    def get_predictions(self):
        text = self.preprocess()
        sequence = self.token.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        inputs = tf.expand_dims(padded[0], 0)
        output = self.model.predict(inputs)
        sentiment = np.argmax(output)
    
        if sentiment == 0:
            print(f'{sentiment}\nThe tweet has positive sentiment.')
        elif sentiment ==1:
            print(f'{sentiment}\nThe tweet has negative sentiment.')
        elif sentiment == 2:
            print(f'{sentiment}\nThe tweet has neutral sentiment.')

