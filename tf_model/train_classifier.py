import pandas as pd
import numpy as np
from time import time
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


glove_vectors = dict()
with open('glove.twitter.27B.200d.txt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:])
        glove_vectors[word] = vector


class CMPredict(tf.Module):

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, ), dtype=tf.string)])
    def prediction(self, tweet: str):

        return {
            'prediction': self.model(tweet)
        }
    
    
class CMModel(Model):

    def __init__(self,
                 num_classes=3,
                 vocab_size=20_000,
                 max_length=100,
                 embed_dim=200,
                 num_filters=256,
                 ffn_units=128,
                 dropout_rate=0.5,
                 batch_size=32,
                 epochs=50,
                 name='covid_mask_classifier'):
        super(CMModel, self).__init__(name=name)
        
        self.wrapper: CMPredict()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.ffn_units = ffn_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.bigram = Conv1D(self.num_filters, 2, activation='relu', name='bigram')
        self.trigram = Conv1D(self.num_filters, 3, activation='relu', name='trigram')
        self.fourgram = Conv1D(self.num_filters, 4, activation='relu', name='fourgram')
        self.fivegram = Conv1D(self.num_filters, 5, activation='relu', name='fivegram')
        self.pool = GlobalMaxPooling1D()
        self.concat = Concatenate()
        self.dense = Dense(self.ffn_units, activation='relu', name='dense')
        self.dropout = Dropout(self.dropout_rate, name='dropout')
        if self.num_classes == 2:
            self.preds = Dense(1, activation='sigmoid', name='predictions')
        else:
            self.preds = Dense(self.num_classes, activation='softmax', name='predictions')
            
        
    def get_data(self):
        
        tweets = pd.read_csv('data.csv')
        data = tweets['tweet'].to_numpy()
        labels = tweets['label'].to_numpy()
        
        return data, labels
        
        
    def data_splits(self, data: np.ndarray, labels: np.ndarray):
        
        train_data, test_data, train_labels, test_labels = train_test_split(
            data,
            labels,
            test_size=0.15,
            random_state=7,
            stratify=labels)
        
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data,
            train_labels,
            test_size=0.15,
            random_state=7,
            stratify=train_labels)
        
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    

    def get_weights(self, dataset: np.ndarray):

        class_weights = compute_class_weight('balanced', np.unique(dataset), dataset)
        class_weights = dict(enumerate(class_weights))

        return class_weights


    def preprocess_tweet(self, tweet: str):
    
        remove_linebreak = tf.strings.regex_replace(tweet, '\n', '')
        remove_tab = tf.strings.regex_replace(remove_linebreak, '\t', '')
        replace_ampersand = tf.strings.regex_replace(remove_tab, '&amp;', 'and')
        remove_url = tf.strings.regex_replace(replace_ampersand, 'http\S+', '')
        remove_mention = tf.strings.regex_replace(remove_url, '@\w+ ', ' ')
        remove_contraction = tf.strings.regex_replace(remove_mention, "'", '')
        alpha_char = tf.strings.regex_replace(remove_contraction, '[^a-zA-Z.]', ' ')
        remove_outer_whitespace = tf.strings.strip(alpha_char)
        remove_extra_whitespace = tf.strings.regex_replace(remove_outer_whitespace, '\s{2,}', ' ')
        
        return tf.strings.lower(remove_extra_whitespace)
    
    
    def create_vectorizer(self, dataset: np.ndarray) -> TextVectorization:
        
        vectorizer = TextVectorization(
            max_tokens=self.vocab_size,
            standardize=self.preprocess_tweet,
            output_mode='int',
            output_sequence_length=self.max_length,
            name='vectorizer')
        
        vectorizer.adapt(dataset)

        return vectorizer
    
    
    def create_emb_matrix(self, vectorizer):
        
        vocab = vectorizer.get_vocabulary()
        word_index = dict(zip(vocab, range(len(vocab))))
        
        emb_matrix = np.zeros((self.vocab_size, self.embed_dim))
        for word, index in word_index.items():
            vector = glove_vectors.get(word)
            if vector is not None:
                emb_matrix[index] = vector
                
        return emb_matrix
        
        
    def call(self, inputs, training=False):
        
        data, labels = self.get_data()
        train_data, train_labels, val_data, val_labels, _, _ = self.data_splits(data, labels)
        vectorizer = self.create_vectorizer(data)
        emb_matrix = self.create_emb_matrix(vectorizer)
        
        tweet_input = Input(shape=(1,), dtype=tf.string, name='tweet')
        
        x = vectorizer(tweet_input)
        
        x = Embedding(input_dim=self.vocab_size,
                      output_dim=self.embed_dim,
                      input_length=self.max_length,
                      weights=[emb_matrix],
                      trainable=False,
                      name='GloVe_200')(x)
        
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)

        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)

        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)

        x_4 = self.fivegram(x)
        x_4 = self.pool(x_4)
        
        merged = self.concat([x_1, x_2, x_3, x_4])
        
        merged = self.dense(merged)
        
        if training:
            merged = self.dropout(merged)

        predictions = self.preds(merged)
        
        model = Model(tweet_input, predictions)

        return model


    def compile_model(self, model):

        if self.num_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
        else:
            model.compile(
            loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    
        return model


    def custom_callbacks(self):

        es = EarlyStopping(
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            patience=7,
            verbose=1
        )

        plateau = ReduceLROnPlateau(
            monitor='val_sparse_categorical_accuracy',
            mode='max',
            factor=0.1,
            patience=3,
            verbose=1
        )

        return es, plateau
    
    
    def train(self):
        all_tweets, all_labels = self.get_data()
        train_data, train_labels, val_data, val_labels, _, _ = self.data_splits(all_tweets, all_labels)
        es, plateau = self.custom_callbacks()
        
        model = self.call(train_data, training=True)

        model = self.compile_model(model)
        
        model.fit(train_data,
                  train_labels,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_data=(val_data, val_labels),
                  class_weight=self.get_weights(train_labels),
                  callbacks=[es, plateau]
                  )

        return model
    
    def wrap_model(self):
        model = self.train()
        self.wrapper = CMPredict(model)
        tf.saved_model.save(
            self.wrapper.model, f'covid_mask_classifier/saved_models/{int(time())}',
            signatures={'serving_default': self.wrapper.prediction}
        )



if __name__ == '__main__':
    model = CMModel()
    model.wrap_model()