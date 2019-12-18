#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:19:28 2018

@author: txqq
"""

#import sys
#sys.path.append("./deepqsar")
#sys.path.append("./deepqsar_ic50")
#from deepqsar import deepscore
#from deepqsar_ic50 import deepscore_IC50



import IC50_predict_generate
from IC50_predict_generate import predict_IC50
import EC50_predict_generate
from EC50_predict_generate import predict_EC50
import os
import numpy as np
np.random.seed(123)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

model_ec50 = EC50_predict_generate
model_ic50 = IC50_predict_generate
def dnn_filter_smiles(number,path,smiles):

    count=0
    dnn_filtered_smiles=[] 

    if not os.path.exists(path):
        os.mkdir(path)
   
    dnnpath=os.path.join(path,'dnn_filter_5rules_smiles'+str(number)+'.txt')
    fout=open(dnnpath,"w")
    for smile in smiles:
        ec50=model_ec50.predict_EC50([smile])
        ic50=model_ic50.predict_IC50([smile])
        if ec50[0][1]>6 and ec50[0][2]>6 and ic50[0][0]>6 and ic50[0][3]>6:
            count=count+1
            dnn_filtered_smiles.append(smile)
            
            fout.write(smile+"\n")
            
    print("dnn_filtered_smiles:%i"%count)
    
    fout.close()
    return dnn_filtered_smiles

