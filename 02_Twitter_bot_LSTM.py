# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:19:28 2017

@author: Rahul
"""
verbose=1
from __future__ import print_function
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances

import numpy as np
import twitter # pip install python-twitter

# Fetching tweets

def get_tweets():
    api = twitter.api(consumer_key='d3t5F3wXmgiyUkbKNJzQo8CmT',
                      consumer_secret='Rklcd7zkTbmOw7X9DS9U5DPwZWjE74HbZhfVsovIblNYgLqgDj',
                      access_token_key='2sVJtXU60KhcEFuE3eXznF8Rn',
                      access_token_secret='l0ZmId6T4PHqK6VolwVQvVt7Io9F424AAF5puduXLKMvs7IG7q')
    tweets = []
    max_id = None
    for _ in range(100):
        tweets.extend(list(api.GetUserTimeline(screen_name='doctorsroom',
                                               max_id=max_id,
                                               count=200,
                                               include_rts=False,
                                               exclude_replies=True)))
        max_id = tweets[-1].id
    return [tweet.text for tweet in tweets]

#np.save(file='./doctorsroom_tweets.npy', arr=get_tweets())
get_tweets=np.load(file='./doctorsroom_tweets.npy')

# Creating the corpus

CORPUS_LENGTH = None

def get_corpus(verbose=0):
    tweets = np.load('./doctorsroom_tweets.npy')
    tweets = [t for t in tweets if 'http' not in t]
    tweets = [t for t in tweets if '&gt' not in t]
    corpus = u' '.join(tweets)
    global CORPUS_LENGTH
    CORPUS_LENGTH = len(corpus)
    if verbose:
        print('Corpus Length:', CORPUS_LENGTH)
    return corpus

corpus = get_corpus(verbose=verbose)

# Chracter index mapping

N_CHARS = None

def create_index_char_map(corpus, verbose=0):
    chars = sorted(list(set(corpus)))
    global N_CHARS
    N_CHARS = len(chars)
    if verbose:
        print('No. of unique characters:', N_CHARS)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char

chars, char_to_idx, idx_to_char = create_index_char_map(corpus, verbose=verbose)

# Sequence creation

MAX_SEQ_LENGTH = 40
SEQ_STEP = 3
N_SEQS = None

def create_sequences(corpus, verbose=0):
    sequences, next_chars = [], []
    for i in range(0, CORPUS_LENGTH - MAX_SEQ_LENGTH, SEQ_STEP):
        sequences.append(corpus[i:i + MAX_SEQ_LENGTH])
        next_chars.append(corpus[i + MAX_SEQ_LENGTH])
    global N_SEQS
    N_SEQS = len(sequences)
    if verbose:
        print('No. of sequences:', len(sequences))
    return np.array(sequences), np.array(next_chars)

sequences, next_chars = create_sequences(corpus, verbose=verbose)

# One hot encoding

def one_hot_encode(sequences, next_chars, char_to_idx):
    X = np.zeros((N_SEQS, MAX_SEQ_LENGTH, N_CHARS), dtype=np.bool)
    y = np.zeros((N_SEQS, N_CHARS), dtype=np.bool)
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1
    return X, y

X, y = one_hot_encode(sequences, next_chars, char_to_idx)

# Create an LSTM model

def build_model(hidden_layer_size=128, dropout=0.2, learning_rate=0.01, verbose=0):
    model = Sequential()
    model.add(LSTM(hidden_layer_size, return_sequences=True, input_shape=(MAX_SEQ_LENGTH, N_CHARS)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden_layer_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(N_CHARS, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
    if verbose:
        print('Model Summary:')
        model.summary()
    return model

model = build_model(verbose=verbose)

# Training a model

def train_model(model, X, y, batch_size=128, nb_epoch=60, verbose=0):
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='loss', verbose=verbose, save_best_only=True, mode='min')
    model.fit(X, y, batch_size=batch_size, epochs=nb_epoch, verbose=verbose, callbacks=[checkpointer])

train_model(model, X, y, verbose=verbose)

# Set random seed

np.random.seed(1337)

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / 0.2
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate a tweet

def generate_tweets(model, corpus, char_to_idx, idx_to_char, n_tweets=10, verbose=0): 
    model.load_weights('weights.hdf5')
    tweets = []
    spaces_in_corpus = np.array([idx for idx in range(CORPUS_LENGTH) if corpus[idx] == ' '])
    for i in range(1, n_tweets + 1):
        begin = np.random.choice(spaces_in_corpus)
        tweet = u''
        sequence = corpus[begin:begin + MAX_SEQ_LENGTH]
        tweet += sequence
        if verbose:
            print('Tweet no. %03d' % i)
            print('=' * 13)
            print('Generating with seed:')
            print(sequence)
            print('_' * len(sequence))
        for _ in range(100):
            x = np.zeros((1, MAX_SEQ_LENGTH, N_CHARS))
            for t, char in enumerate(sequence):
                x[0, t, char_to_idx[char]] = 1.0

            preds = model.predict(x, verbose=0)[0]
            next_idx = sample(preds)
            next_char = idx_to_char[next_idx]

            tweet += next_char
            sequence = sequence[1:] + next_char
        if verbose:
            print(tweet)
            print()
        tweets.append(tweet)
    return tweets

tweets = generate_tweets(model, corpus, char_to_idx, idx_to_char, verbose=verbose)

# Evaluate the model

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sequences)
Xval = vectorizer.transform(tweets)
print(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(axis=1).mean())