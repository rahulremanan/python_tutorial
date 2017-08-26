import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, concatenate, dot, TimeDistributed, Flatten, Merge
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
import argparse
import time
import datetime as dt
import os
import glob

class housekeeping:
    def is_valid_file(parser, arg):
        arg = arg.replace("'",'' )
        arg = arg.replace('"', '')
        if not os.path.isfile(arg):
            parser.error("the file %s does not exist..." % arg)
        else:
            return arg
    def is_valid_dir(parser, arg):
        arg = arg.replace("'",'' )
        arg = arg.replace('"', '')
        if not os.path.isdir(arg):
            parser.error("the folder %s does not exist..." % arg)
        else:
            return arg
    def mkdate(datestr):
        datestr = datestr.replace("'","")
        datestr = datestr.replace('"','')
        try:
            datestr = dt.datetime.strptime(datestr, '%Y,%m,%d')
        except:
            datestr = dt.datetime.strptime(datestr, '%Y-%m-%d')
        return datestr
    def to_integer(dt_time):
        return dt_time.year, dt_time.month, dt_time.day
    def generate_timestamp():
        timestamp=0
        try:
            if timestamp <1:
                timestamp +=1
                timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
                print ("time stamp generated: "+timestring)
                return timestring
            else:
                timestamp+=1
                print ("time stamp already generated: "+timestring)
        except:
            print ("error generating time stamp...")    
    def create_data_dir(directory):
         if not os.path.exists(directory):
            os.makedirs(directory)

train_model = 0
train_epochs = 66
load_model = 1
batch_size = 750
lstm_size = 32
test_qualitative = 0
user_questions = 1
num_recurrent_units = 8
dropout = 0.3
timestr = housekeeping.generate_timestamp()
dataset_loc = ('/home/info/babi_tasks_1-20_v1-2.tar.gz')
weights_loc = ('/home/info/qa_demo_weights*.[Hh]5')
stateful_weights_loc = '/home/info/qa_demo_weights_stateful*.[Hh]5'
weights_save_loc = ('/home/info/qa_demo_weights_'+timestr+'.h5')
stateful_weights_save_loc = ('/home/info/qa_demo_weights_stateful_'+timestr+'.h5')

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

#try:
#    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://github.com/rahulremanan/python_tutorial/blob/master/NLP/data/babi_tasks_1-20_v1-2.tar.gz')
#except:
#    print('Error downloading dataset, please download it manually:\n'
#          '$ wget https://github.com/rahulremanan/python_tutorial/blob/master/babi_tasks_1-20_v1-2.tar.gz'
#          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi_tasks_v1-2.tar.gz')
#    raise
tar = tarfile.open(dataset_loc)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
idx_word = dict((i+1, c) for i,c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

print('Input sequence:', input_sequence)
print('Question:', question)

input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(dropout))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(dropout))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
question_encoder.add(Dropout(dropout))

input_encoded_m = input_encoder_m(input_sequence)
print('Input encoded m', input_encoded_m)
input_encoded_c = input_encoder_c(input_sequence)
print('Input encoded c', input_encoded_c)
question_encoded = question_encoder(question)
print('Question encoded', question_encoded)

match = dot([input_encoded_m, question_encoded], axes=(2, 2), normalize=False)
match = Activation('softmax')(match)
print('Match shape', match)

response = add([match, input_encoded_c]) 
response = Permute((2, 1))(response) 
print('Response shape', response)

answer = concatenate([response, question_encoded])
print('Answer shape', answer)

if load_model ==1:
    try:
        newest_weights = max(glob.iglob(weights_loc), 
                         key=os.path.getctime)
        newest_weights_stateful = max(glob.iglob(stateful_weights_loc), 
                                  key=os.path.getctime)
        print (newest_weights, newest_weights_stateful)
    except:
        print ('Error loading saved weights...')

class RNN:
    def LSTM_layer(input_sequence, question, answer, lstm_size, dropout, vocab_size, batch_size):
        answer = LSTM(units=lstm_size, return_sequences=True)(answer)
        answer = Dropout(dropout)(answer)
        for cur_unit in range(num_recurrent_units):
            answer = LSTM(lstm_size, return_sequences=True)(answer)
        answer = LSTM(lstm_size)(answer)
        answer = Dropout(dropout)(answer)
        answer = Dense(vocab_size)(answer)
        answer = Activation('softmax')(answer)
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
        return model

try:
    model = RNN.LSTM_layer(input_sequence, question, answer, lstm_size, dropout, vocab_size, batch_size)
    model.summary()
    if load_model == 1:
        model = keras.models.load_model(newest_weights)
    if train_model == 1:
        model.fit([inputs_train, queries_train], answers_train, batch_size, train_epochs,
          validation_data=([inputs_test, queries_test], answers_test))
        model.save(weights_save_loc)
except:
    print ("Can't train the model...")

if test_qualitative == 1:
    print('-------------------------------------------------------------------------------------------')
    print('Qualitative Test Result Analysis')
    for i in range(0,10):
        current_inp = test_stories[i]
        current_story, current_query, current_answer = vectorize_stories([current_inp], word_idx, story_maxlen, query_maxlen)
        current_prediction = model.predict([current_story, current_query])
        current_prediction = idx_word[np.argmax(current_prediction)]
        print(' '.join(current_inp[0]), ' '.join(current_inp[1]), '| Prediction:', current_prediction, '| Ground Truth:', current_inp[2])

if user_questions == 1:
    print('-------------------------------------------------------------------------------------------')
    print('Custom User Queries (Make sure there are spaces before each word)')
    while 1:
        try:
            print('-------------------------------------------------------------------------------------------')
            print('Please input a story')
            user_story_inp = input().split(' ')
            print('Please input a query')
            user_query_inp = input().split(' ')
            user_story, user_query, user_ans = vectorize_stories([[user_story_inp, user_query_inp, '.']], word_idx, story_maxlen, query_maxlen)
            user_prediction = model.predict([user_story, user_query])
            user_prediction = idx_word[np.argmax(user_prediction)]
            print('Result')
            print(' '.join(user_story_inp), ' '.join(user_query_inp), '| Prediction:', user_prediction)
        except:
            print ('Question cannot be understood. Please ensure spaces between each words...')
