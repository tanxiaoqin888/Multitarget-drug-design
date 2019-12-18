#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:05:33 2018

@author: txqq
"""

import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
import pickle
import os
import tensorflow as tf
from keras.layers.recurrent import LSTM
import pickle
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth= True
set_session(tf.Session(config=config))
from sklearn.metrics import precision_score

# load the training smiles file
input_file="/home/txqq/work/datamake/copy_degree_data/IC50_EC50_activity_multi_activity_3d_guiyihua_copy.txt"

#input_file1 = 'zinc_smiles.txt'
text = open(input_file, 'r').read()
train_text = text[:int(len(text) * 0.8)]
test_text = text[int(len(text) * 0.8) + 1:]

# index the chars
char_to_idx= {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)



# save the index-dicts for reuse during generation

char_to_idx=pickle.load(open('char_to_idx.p','rb'))
idx_to_char=pickle.load(open('idx_to_char.p', 'rb'))
vocab_size = len(char_to_idx)
#print('Working on %d characters (%d unique)' % (len(train_text), vocab_size))
# model hyperparameters
SEQ_LEN = 64
BATCH_SIZE = 128
EPOCH = 60
BATCH_CHARS = len(train_text) // BATCH_SIZE // SEQ_LEN

test_steps = len(test_text) // BATCH_SIZE // SEQ_LEN
test_x = test_text[0:test_steps * SEQ_LEN * BATCH_SIZE]
test_y = test_text[1:test_steps * SEQ_LEN * BATCH_SIZE + 1]
test_x = np.asarray([char_to_idx[c] for c in test_x], dtype=np.int32)
test_y = np.asarray([char_to_idx[c] for c in test_y], dtype=np.int32)
X_T = np.zeros((test_steps * BATCH_SIZE, SEQ_LEN, vocab_size))
Y_T = np.zeros((test_steps * BATCH_SIZE, SEQ_LEN, vocab_size))
for i in range(0, test_steps):
    for j in range(0, BATCH_SIZE):
        start = BATCH_SIZE * SEQ_LEN * i + SEQ_LEN * j
        for k in range(0, SEQ_LEN):
            X_T[BATCH_SIZE * i + j, k, test_x[start + k]] = 1
            Y_T[BATCH_SIZE * i + j, k, test_y[start + k]] = 1
test_x = X_T
test_y = Y_T

LSTM_SIZE = 1024
LAYERS = 2

# For training, each subsequent example for a given batch index should be a
# consecutive portion of the text.  To achieve this, each batch index operates
# over a disjoint section of the input text.
def read_batches(text):
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    X = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    Y = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))

    #yield X, Y
    for i in range(0, BATCH_CHARS):  #the i BATCH_SIZE*SEQ_LEN block
        X[:] = 0  #initialization
        Y[:] = 0
        for j in range(0, BATCH_SIZE):  #the j SEQ_LEN block
            start = BATCH_SIZE * SEQ_LEN * i + SEQ_LEN * j
            for k in range(0, SEQ_LEN):  #the k sample
                X[j, k, T[start + k]] = 1
                Y[j, k, T[start + k + 1]] = 1

        yield X, Y


def build_model(infer):
    if infer:
        batch_size = 1
        seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(
        LSTM(
            LSTM_SIZE,
            return_sequences=True,
            batch_input_shape=(batch_size, seq_len, vocab_size),
            stateful=False))

    model.add(Dropout(0.2))
    for l in range(LAYERS):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=False))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001, clipnorm=5)
    model.compile(
        loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


print('Building model.')
training_model = build_model(infer=False)
training_model.load_weights('./modelseps/keras_char_rnn.16.h5')
test_model = build_model(infer=True)
print('... done')


def sample(epoch, sample_chars=512, primer_text='\n'):
    test_model.reset_states()
    test_model.load_weights('./fine_tune_model/keras_char_rnn.%d.h5' % epoch)
    sampled = [char_to_idx[c] for c in primer_text]

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()
        sample = np.random.choice(range(vocab_size), p=softmax)
        sampled.append(sample)
        #print(len(sampled))

    print(''.join([idx_to_char[c] for c in sampled]))


#training_model.load_weights('model_weights.h5')

inds = 0
test_YT = np.zeros((test_steps * SEQ_LEN * BATCH_SIZE, vocab_size))
for i in range(test_y.shape[0]):
    for j in range(test_y.shape[1]):
        test_YT[inds, :] = test_y[i, j, :]
        inds += 1
test_YT = np.argmax(
    test_YT, axis=1).reshape(test_steps * SEQ_LEN * BATCH_SIZE, 1)

print('Start training...')
if not os.path.exists('./new2_fine_tune_model_60epoch_guiyihua'):
    os.mkdir("./new2_fine_tune_model_60epoch_guiyihua")

bestloss = [100.0, -1.0]
for epoch in range(1, EPOCH + 1):
    for i, (x, y) in enumerate(read_batches(train_text)):
        loss = training_model.train_on_batch(x, y)

        # reset the rnn states every 100 mini-batches
        if i % 100 == 0:
            training_model.reset_states()
            loss0, loss1 = training_model.evaluate(
                test_x, test_y, batch_size=BATCH_SIZE, verbose=False)
            mes = str(epoch) + ',' + str(i) + ',' + str(loss0) + ',' + str(
                loss1) + '\n'
            fid = open(os.path.join('./new2_fine_tune_model_60epoch_guiyihua','train_model.csv'), 'a')
            fid.write(mes)
            print(loss0, loss1)
            fid.close()
            if loss0 < bestloss[0] and loss1 > bestloss[1]:
                bestloss[0] = loss0
                bestloss[1] = loss1
                training_model.save_weights('./new2_fine_tune_model_60epoch_guiyihua/model_weights_fine_tune_29epoch.h5', overwrite=True)
    # save weights after each epoch and write msgs
    pred_y = training_model.predict(test_x, batch_size=BATCH_SIZE)
   
    pred_YT = np.zeros((test_steps * SEQ_LEN * BATCH_SIZE, vocab_size))
    inds = 0
    for i in range(pred_y.shape[0]):
        for j in range(pred_y.shape[1]):
            pred_YT[inds, :] = pred_y[i, j, :]
            inds += 1
    pred_YT = np.argmax(
        pred_YT, axis=1).reshape(test_steps * SEQ_LEN * BATCH_SIZE, 1)
    prec = precision_score(test_YT, pred_YT, average='micro')
    mes = str(epoch) + ',' + str(prec) + '\n'
    fid = open('./new2_fine_tune_model_60epoch_guiyihua/train_model_precision_score.csv', 'a')
    fid.write(mes)
    fid.close()
   
    training_model.save_weights(
        './new2_fine_tune_model_60epoch_guiyihua/keras_char_rnn_epoch.%d.h5' % epoch, overwrite=True)
