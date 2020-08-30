import pandas as pd
import numpy as np
import re
from num2words import num2words
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Sequential, LSTM, Bidirectional, Dense, Dropout

tweet = pd.read_csv('tweet.csv')
tweet['label'] = tweet['label'].map({'positive': 0, 'negative': 1, 'neutral': 2})
tweets = tweet['tweet'].to_list()
labels = tweet['label'].to_list()

def preprocess(text):
    
    def remove_URL(text):
        text = re.sub(r'http\S+', '', text)
        return text
    
    def remove_mentions(text):
        text = re.sub(r'@\S+', '', text)
        return text
    
    def remove_next_page(text):
        text = text.replace('\n', '  ')
        return text
    
    def remove_apostrophes(text):
        text = text.replace("'", '')
        return text
    
    def convert_nums(text):
        for num in re.findall(r'(?:\d+)(?:\,\d+)*(?:\,\s{1}\d{3})*(?:\s{1}\d{3})*(?:\.\d+)*', text):
            new_num = num.replace(' ', '').replace(',', '')
            num_text = ' ' + num2words(new_num) + ' '
            text = text.replace(num, num_text)
        return text
    
    def split_words(text):
        text = re.findall(r"[\w]+|!|\"|#|\$|%|&|\(|\)|\*|\+|,|-|\.|/|:|;|<|>|@|\[|\]|\\|\^|`|{|}|~", text)
        return text
    
    text = remove_URL(text)
    text = remove_mentions(text)
    text = remove_next_page(text)
    text = remove_apostrophes(text)
    text = convert_nums(text)
    text = split_words(text)
    
    return text

clean_tweets = []
for tweet in tweets:
    clean_tweets.append(preprocess(tweet))

token = Tokenizer(filters='\t\n')
token.fit_on_texts(clean_tweets)
vocab_size = len(token.word_index) + 1
sequences = token.texts_to_sequences(text)
max_length = 120
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

glove_vectors = dict()
file = open('glove.twitter.27B/glove.twitter.27B.200d.txt', encoding='utf-8')
for line in file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:])
    glove_vectors[word] = vector
file.close()

word_vector_matrix = np.zeros((vocab_size, 200))
for word, index in token.word_index.items():
    vector = glove_vectors.get(word)
    if vector is not None:
        word_vector_matrix[index] = vector

model = Sequential()
model.add(Embedding(vocab_size, vec_size, input_length=max_length, weights=[word_vector_matrix], trainable=False))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

