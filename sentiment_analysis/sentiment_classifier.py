import pandas as pd
import numpy as np
from sklearn.model_selection import test_train_split


# load df

X = df['tweet']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_tweets)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_tweets)
train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding='post', truncating='post'
)

test_sequences = tokenizer.texts_to_sequences(test_tweets)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post'
)

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras initializaers import constant
from keras.optimizers import Adam

model = Sequential()

model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    train_padded, train_labels, epoch=10, validation_data=(test_padded, test_labels)
)







