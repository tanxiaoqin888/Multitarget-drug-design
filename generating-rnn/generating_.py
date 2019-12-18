import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
import pickle
import rdkit
from rdkit import Chem
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth= True
set_session(tf.Session(config=config))
# loads the smiles symbol dicts
char_to_idx = pickle.load(open('char_to_idx.p', 'rb'))
idx_to_char = pickle.load(open('idx_to_char.p', 'rb'))

vocab_size = len(char_to_idx)

# MODEL HYPERPARAMETERS
LSTM_SIZE = 1024
LAYERS = 2

#Number of molecules to sample in parallel
BANK_SIZE = 100


def build_generator_model():
    seq_len = 1
    model = Sequential()
    model.add(
        LSTM(
            LSTM_SIZE,
            return_sequences=True,
            batch_input_shape=(BANK_SIZE, seq_len, vocab_size),
            stateful=True))

    model.add(Dropout(0.2))
    for l in range(LAYERS):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))

    opt = Adam(lr=0.001, clipnorm=5)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model


print('Building model.')
test_model = build_generator_model()


def sample(sample_chars=10 * 50, primer_text='\n'):
    test_model.reset_states()
    samplebank = []
    sampled = [char_to_idx[c] for c in primer_text]
    for i in range(BANK_SIZE):
        samplebank.append(sampled[:])

    for _ in range(sample_chars):
        batch = np.zeros((BANK_SIZE, 1, vocab_size))
        for k, bank in enumerate(samplebank):
            batch[k, 0, bank[-1]] = 1
        print(samplebank)
        softmax = test_model.predict_on_batch(batch)
        for k in range(BANK_SIZE):
            sample = np.random.choice(range(vocab_size), p=softmax[k].ravel())
            samplebank[k].append(sample)


# prints out the sampled molecules. alternatively, you can write to file here.
#for bank in samplebank:
#print(''.join([idx_to_char[c] for c in bank]))
    return samplebank

test_model.load_weights('/home/txqq/work/RNN/lstm5/filter_smiles/data/true_model/nova/epoch16_50/keras_char_rnn_epoch.25.h5')

print("sampling %i in parallel" % BANK_SIZE)

samplebank = sample(sample_chars=180 * 50)

valid_samples = 0
all_samples = 0
fid = open('/home/txqq/work/RNN/lstm5/filter_smiles/data/true_model/nova/lstm_big_data_2018_11_15_25.txt', 'w')
for bank in samplebank:
    test_samples = ''.join([idx_to_char[c] for c in bank])
    for sample in test_samples.split('\n'):
        if len(sample) <= 1:
            continue
        print(sample)
        all_samples += 1
        sample_valid = Chem.MolFromSmiles(sample)
        if sample_valid != None:
            valid_samples += 1
            fid.write(sample + '\n')
fid.write(str(valid_samples)+','+str(all_samples)+','+str(valid_samples / all_samples))
print(valid_samples, all_samples, valid_samples / all_samples)
fid.close()

