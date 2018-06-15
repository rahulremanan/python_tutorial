# -*- coding: utf-8 -*-
#"""
#Created on Tue Jul 15 18:21:28 2018
#
#@author: Dr. Rahul Remanan
#@email: info@moad.computer
#"""
from __future__ import print_function
verbose = False

import os 
import random

import keras

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, \
                         Activation,\
                         Dropout, \
                         LSTM, \
                         GRU, \
                         Bidirectional, \
                         TimeDistributed, \
                         BatchNormalization
from keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances


import numpy as np
import twitter # pip install python-twitter

# Fetching tweets

def get_tweets():
    consumer_key='Enter consumer key'
    consumer_secret='Enter consumer secret'
    access_token_key='Enter access token key'
    access_token_secret='Enter access token secret'
    
    api = twitter.Api(consumer_key,
                      consumer_secret,
                      access_token_key,
                      access_token_secret)
    
    tweets = []
    max_id = None
    for _ in range(100):
        tweets.extend(list(api.GetUserTimeline(screen_name='doctorsroom',
                                               max_id=max_id,
                                               count=2000,
                                               include_rts=True,
                                               exclude_replies=True)))
        max_id = tweets[-1].id
    return [tweet.text for tweet in tweets]

save_tweets = True
if save_tweets:
    np.save(file='./doctorsroom_tweets.npy', arr=get_tweets())    
    
load_tweets = False
if load_tweets:
  get_tweets=np.load(file='./doctorsroom_tweets.npy')
else:
    get_tweets = get_tweets()

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

batch_size=5

def build_model(unit_size=34, 
                dropout=0.75, 
                learning_rate=1e-4, 
                verbose=False,
                enable_dropout = True,
                enable_batchnormalization = False,
                layer_depth = 200,
                activation = 'relu',
                batch_size = batch_size):
    model = Sequential()
    model.add(Bidirectional(GRU(unit_size, 
                                 return_sequences=True), 
                                 input_shape=(MAX_SEQ_LENGTH, N_CHARS), batch_size = batch_size))
    model.add(TimeDistributed(Dropout(dropout)))
    for i in range(layer_depth):
      model.add(Bidirectional(GRU(units= unit_size,
                return_sequences=True,
                activation=activation,
                stateful=False)))
      if enable_dropout == True:
        model.add(Dropout(dropout))
      if enable_batchnormalization == True:
        model.add(TimeDistributed(BatchNormalization()))
    
    model.add(GRU(units= unit_size,
              return_sequences=False,
              activation=activation,
              stateful=False))
    
    if enable_dropout == True:
      model.add(Dropout(dropout))
    
    model.add(Dense(units=4096, 
                    activation=activation,
                    use_bias=True))
    model.add(Dropout(dropout))
    model.add(Dense(N_CHARS, activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=RMSprop(lr=learning_rate))
    if verbose:
        print('Model Summary:')
        model.summary()
    return model

model = build_model(verbose=verbose)

checkpointer_path = './weights.hdf5'
load_checkpointer = False

if os.path.exists(checkpointer_path) and load_checkpointer:
  model.load_weights(checkpointer_path)

# Training a model

def train_model(model, X, y, batch_size=batch_size, nb_epoch=60, verbose=0):
    checkpointer = ModelCheckpoint(filepath=checkpointer_path, 
                                   monitor='loss', 
                                   verbose=verbose, 
                                   save_best_only=True, 
                                   mode='min')
    early_stopper = EarlyStopping(monitor='loss', 
                                  min_delta=0, 
                                  patience=4, 
                                  verbose=0, 
                                  mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='loss', 
                                  factor=0.5, 
                                  patience=1, 
                                  min_lr=1e-16, 
                                  verbose=1)
    model.fit(X, y, 
              batch_size=batch_size, 
              epochs=nb_epoch, 
              verbose=verbose, 
              callbacks=[checkpointer, 
                         early_stopper])

train_model(model, X, y, verbose=verbose)
# Set random seed

np.random.seed(random.randint(1, 10**4))

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / 0.2
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate a tweet

# Generate a tweet

def generate_tweets(model, 
                    corpus, 
                    char_to_idx, 
                    idx_to_char, 
                    n_tweets=10,
                    batch_size = batch_size,
                    use_bidirectional = True,
                    verbose=0): 
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
            if use_bidirectional:
              x = np.resize(x, (batch_size, x.shape[1], x.shape[2]))
            preds = model.predict(x, batch_size = None, verbose=0, steps = batch_size)[0]
            next_idx = sample(preds)
            next_char = idx_to_char[next_idx]

            tweet += next_char
            sequence = sequence[1:] + next_char
        if verbose:
            print(tweet)
            print()
        tweets.append(tweet)
    return tweets

tweets = generate_tweets(model, 
                         corpus, 
                         char_to_idx, 
                         idx_to_char, 
                         verbose=verbose)

# Evaluate the model

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(sequences)
Xval = vectorizer.transform(tweets)
print(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(axis=1).mean())