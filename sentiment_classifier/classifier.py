import pandas as pd
import numpy as np
import re
import json
import io
import assets

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = pd.read_csv('anno.csv')
data = data[data['label'].notnull()].reset_index()
data = data[['tweet', 'label']]
sent_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
data['label'] = data['label'].map(sent_dict).astype(int)

tweets = data.tweet.to_list()
labels = data.label.to_list()

class_weights = compute_class_weight('balanced', np.unique(labels), list(labels))
class_weights = dict(enumerate(class_weights))

def preprocess(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = tweet.replace("'", '')
    tweet = tweet.lower()
    tweet = re.sub(r'[^a-zA-Z.?!]', ' ', tweet) 
    tweet = re.sub(r'\s{2,}', ' ', tweet)
    tweet = tweet.strip()
    
    return tweet

clean_tweets = []
for tweet in tweets:
    clean_tweets.append(preprocess(tweet))


token = Tokenizer()
token.fit_on_texts(clean_tweets)
VOCAB_SIZE = len(token.word_index) + 1
sequences = token.texts_to_sequences(clean_tweets)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

X = padded
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=313,
    stratify=labels
    )

glove_vectors = dict()
with open('glove.twitter.27B.200d.txt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:])
        glove_vectors[word] = vector


word_vector_matrix = np.zeros((VOCAB_SIZE, EMB_DIM))
for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector

model = Sequential()

model.add(Embedding(input_dim=VOCAB_SIZE,
                    output_dim=EMB_DIM,
                    input_length=MAX_LENGTH,
                    weights=[word_vector_matrix],
                    trainable=False))

model.add(Conv1D(FILTER, 2, activation='relu'))
model.add(Conv1D(FILTER, 3, activation='relu'))
model.add(Conv1D(FILTER, 4, activation='relu'))
model.add(Conv1D(FILTER, 5, activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(FFN, activation='relu'))

model.add(Dropout(DROPOUT))

model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy',
             optimizer=Adam(learning_rate=0.0001),
             metrics=['categorical_accuracy']
            )


model_callbacks = [
    EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=10, verbose=1),
    ModelCheckpoint('new_mask', monitor='val_categorical_accuracy', mode='max', save_best_only=True, verbose=1)
]


model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NB_EPOCHS,
    validation_split=0.2,
    shuffle=False,
    class_weight=class_weights,
    callbacks=model_callbacks
    )


model.save('covid_mask', save_format='h5')

token_json = token.to_json()
with io.open('token.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(token_json, ensure_ascii=False))