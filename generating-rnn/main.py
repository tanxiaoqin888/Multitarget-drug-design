#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:15:58 2019

@author: txqq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:21:37 2018

@author: txqq
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
import pickle
from rdkit import Chem
import os
from filter_smiles_5rule import filter_generate_smiles
from evaluate import evaluate_smiles
from dnn_filtered_smiles_3 import dnn_filter_smiles
from lstm_training import lstm_training
import tidu
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import keras.backend.tensorflow_backend as KTF
#def get_session(gpu_fraction=0.3):
#    num_threads=os.environ.get('OMP_NUM_THREADS')
#    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#    if num_threads:
#        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,intra_op_parallelism_threads=num_threads))
#    else:
#        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#    
#KTF.set_session(get_session())
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth= True
#set_session(tf.Session(config=config))
##import os
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
#os.environ["CUDA_VISIBLE_DEVICES"]="1" 
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#session = tf.Session(config=config)
#KTF.set_session(session )
#%%
#CONTROL_SMILES
control_smiles = []
control=open('/home/txqq/work/dnn/multi_smiles3_nosim3.csv','r')
control_s = control.read()
for i in control_s:
    smiles=i.strip()
    control_smiles.append(smiles)
# loads the smiles symbol dicts
char_to_idx = pickle.load(open('/home/txqq/work/RNN/lstm5/char_to_idx.p', 'rb'))
idx_to_char = pickle.load(open('/home/txqq/work/RNN/lstm5/idx_to_char.p', 'rb'))

vocab_size = len(char_to_idx)

#TRAINING_SMILES
#input_file="/home/txqq/work/RNN/lstm5/filter_smiles/new_data/four_nosim_keku2.csv"
input_file="/home/txqq/work/dnn/multi_smiles3_nosim3.csv"
#input_file="/home/txqq/work/RNN/lstm5/filter_smiles/new_data/four_nosim_caco_chuli3.csv"
text = open(input_file, 'r').read()
train_text = text[:int(len(text) * 0.8)]
test_text = text[int(len(text) * 0.8) + 1:]

#char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
#idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
#vocab_size = len(char_to_idx)
# MODEL HYPERPARAMETERS
LSTM_SIZE = 1024
LAYERS = 2

#Number of molecules to sample in parallel
BANK_SIZE = 100
#prior_f="/home/txqq/work/RNN/lstm5/filter_smiles/data_new/1_16_20_yu_200_2"
prior_f='/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2'
if not os.path.exists(prior_f):
    os.mkdir(prior_f)
valid_smile_f="/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2/multi_yu_ec50_20_9.01.csv"

restore_from=None
f=open( valid_smile_f,"a")
s1='epoch'+ ','+'valid_rate'+','+'nodup_valid_rate'+','+'nodup'+','+'total'+','+'num_5rules_smiles'+','+'num_dnn_smiles'+','+'very_similiar_rate'+','+'mean_pair_similiar'+','+'control_rate_8'+','+'control_mean_similiar'

a=[0,1,2,3,4,5,6,7]
b=["c_chain_is"+str(i)+"rate" for i in a]
b.append("c_chain_length_>7")
s2=",".join(b)
s3=s1+s2+"\n"
f.write(s3)
f.close()
#%%
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
def get_smiles(path):
    smiles=[]
   
    test_model.load_weights(path)
    
    print("sampling %i in parallel" % BANK_SIZE)
    
    samplebank = sample(sample_chars=200 * 50)
    
    valid_samples = 0
    all_samples = 0
    
    for bank in samplebank:
        test_samples = ''.join([idx_to_char[c] for c in bank])
        for s in test_samples.split('\n'):
            if len(s) <= 1:
                continue
            else:
                smiles.append(s)
    return smiles
#%%
save_path='/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2/model'
lstm_save_path = '/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2/5rules_smiles'
dnn_lstm_save_path = '/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2/dnn_smiles'
original_path = '/home/hdd2/lost_data/data_new/2019_2_12_19_yu_200_copy_nodup2/original_smiles'
if not os.path.exists(original_path):
    os.mkdir(original_path)
yu=6.0
for number in range(1,26):
#    path="/home/txqq/work/RNN/lstm5/new3_fine_tune_model_epoch60_guiyihua_all/keras_char_rnn_epoch."+str(number)+".h5"
##    path = '/home/txqq/work/RNN/lstm5/model_test/keras_char_rnn.'+str(number)+'.h5'
    if number==1:
        training_smiles = text
        training_path = '/home/txqq/work/RNN/lstm5/modelseps/keras_char_rnn.16.h5'
#        training_path = '/home/txqq/work/RNN/lstm5/filter_smiles/data/true_model/nova/epoch16_50/keras_char_rnn_epoch.13.h5'
        lstm_training(number,training_smiles,path=save_path,training_path=training_path)
        smiles_path = os.path.join(save_path,'keras_char_rnn_epoch.%d.h5' % number)
        smiles=get_smiles(smiles_path)
        valid_rate,nodup_valid_rate,sum_nodup_smiles,total_smiles,valid_smiles_list=evaluate_smiles().valid_smiles(smiles)
        
        smiles_file = os.path.join(original_path,'original_smiles.%d.txt' % number)
        fout=open(smiles_file,'w')
        for i in valid_smiles_list:
            fout.write(i+'\n')
        fout.close()
        filtered_5rules_smiles = filter_generate_smiles(number,valid_smiles_list,path=lstm_save_path)
        num_filtered_5rules_smiles = len(filtered_5rules_smiles)
        num_filtered_5rules_smiles = str(num_filtered_5rules_smiles)
        
       
        dnn_filtered_5rules_smiles = dnn_filter_smiles(number,path=dnn_lstm_save_path,smiles=filtered_5rules_smiles,s=yu)
        num_dnn_filtered_5rules_smiles = len(dnn_filtered_5rules_smiles)
        num_dnn_filtered_5rules_smiles = str(num_dnn_filtered_5rules_smiles)
        
        similarp = evaluate_smiles().pair_similiar_filter(dnn_filtered_5rules_smiles,control_smiles)
        very_smiliar_rate_8,mean_pair_similiar_control = similarp
        very_similiar_rate,mean_pair_similiar=evaluate_smiles().pair_similiar(dnn_filtered_5rules_smiles)
    #    very_similiar_rate_fcfp,mean_pair_similiar_fcfp=evaluate_smiles().pair_similiar_fcfp4(valid_smiles_list)    
        dict2=evaluate_smiles().length_of_chian_distirbution(dnn_filtered_5rules_smiles)
        f=open( valid_smile_f,"a")
        write_line=[str(number),str(valid_rate),str(nodup_valid_rate),str(sum_nodup_smiles),str(total_smiles),
                    str(num_filtered_5rules_smiles) ,str(num_dnn_filtered_5rules_smiles),
                    str(very_similiar_rate),str(mean_pair_similiar),str(very_smiliar_rate_8),
                    str(mean_pair_similiar_control)
                    ]
        ff=[str(dict2[i]) for i in dict2.keys()]
        write_line.extend(ff)
        f.write(",".join(write_line)+"\n")
        f.close()
      
       
    else:
        dnn_smiles_path = os.path.join(dnn_lstm_save_path,'dnn_smiles_activity'+str(number-1)+'.csv')
        tidu_smiles=tidu.new_smiles(dnn_smiles_path)
        dnn_smiles_char = open(tidu_smiles,'r').read()
#        new_training_char = ''.join(dnn_smiles_char+text)
        new_training_char = ''.join(dnn_smiles_char)
        training_path = os.path.join(save_path,'keras_char_rnn_epoch.%d.h5' % (number-1))
#        os.system("/bin/bash training.sh")
        lstm_training(number,new_training_char,path=save_path,training_path=training_path)
        smiles_path = os.path.join(save_path,'keras_char_rnn_epoch.%d.h5' % number)
        smiles=get_smiles(smiles_path)
        valid_rate,nodup_valid_rate,sum_nodup_smiles,total_smiles,valid_smiles_list=evaluate_smiles().valid_smiles(smiles)
        smiles_file = os.path.join(original_path,'original_smiles.%d.txt' % number)
        fout=open(smiles_file,'w')
        for i in valid_smiles_list:
            fout.write(i+'\n')
        fout.close()
        
        filtered_5rules_smiles = filter_generate_smiles(number,valid_smiles_list,path=lstm_save_path)
        num_filtered_5rules_smiles = len(filtered_5rules_smiles)
        num_filtered_5rules_smiles = str(num_filtered_5rules_smiles)
        
#        os.system("/bin/bash dnn_filter.sh")
        dnn_filtered_5rules_smiles = dnn_filter_smiles(number,path=dnn_lstm_save_path,smiles=filtered_5rules_smiles,s=yu)
        num_dnn_filtered_5rules_smiles = len(dnn_filtered_5rules_smiles)
        num_dnn_filtered_5rules_smiles = str(num_dnn_filtered_5rules_smiles)
    
        
        
        similarp = evaluate_smiles().pair_similiar_filter(dnn_filtered_5rules_smiles,control_smiles)
        very_smiliar_rate_8,mean_pair_similiar_control = similarp
        very_similiar_rate,mean_pair_similiar=evaluate_smiles().pair_similiar(dnn_filtered_5rules_smiles)
    #    very_similiar_rate_fcfp,mean_pair_similiar_fcfp=evaluate_smiles().pair_similiar_fcfp4(valid_smiles_list)    
        dict2=evaluate_smiles().length_of_chian_distirbution(dnn_filtered_5rules_smiles)
        
        f=open( valid_smile_f,"a")
        write_line=[str(number),str(valid_rate),str(nodup_valid_rate),str(sum_nodup_smiles),str(total_smiles),
                    str(num_filtered_5rules_smiles) ,str(num_dnn_filtered_5rules_smiles),
                    str(very_similiar_rate),str(mean_pair_similiar),str(very_smiliar_rate_8),
                    str(mean_pair_similiar_control)
                    ]
        ff=[str(dict2[i]) for i in dict2.keys()]
        write_line.extend(ff)
        f.write(",".join(write_line)+"\n")
        f.close()
        yu=yu+0.1
