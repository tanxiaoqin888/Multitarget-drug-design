#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:33:54 2018

@author: txqq
"""


import pandas as pd
import deepchem as dc
import numpy as np
import gpcr_dataset
import HyperOPTwithEarlyStop6
import get_datapoint
import mk
import os
from t4 import  target_4_save
#%%
import json
from  collections import OrderedDict

#%%
np.random.seed(123)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_file = './ec50_data/ic50_dnn_all_3d_remove_bad_point2.csv'
data_file_,_=os.path.splitext(data_file)



m, all_datasets, transformers = gpcr_dataset.load_gpcr(dataset_file=data_file,
                                                    featurizer='ECFP',
                                                    sep='Harmonious_positive',reload=True)

dataset, train_dataset, valid_dataset, test_dataset = all_datasets





mm=list(get_datapoint.get_datapoints(train_dataset,valid_dataset,test_dataset))
c=dict(tn=mm[0],tra=mm[1],val=mm[2],tes=mm[3],sum_tra_val_tes=mm[4])
print(c)
#the number of the posive samples
d=pd.DataFrame(c,columns=['tn','tra','val','tes','sum_tra_val_tes'])
d.to_csv(os.path.join(data_file_,'./number_quan.csv'))

#%%
# Fit models

import numpy as np



params_dict=OrderedDict([
        ("batch_size", [64]),
        ("learning_rate", [0.00001,0.0001,0.001,0.01]),
        ("weight_init_stddevs",[.033]),
        ("bias_init_consts",[0.1]),
        ("dropouts",[0.5]),
        ("layer_sizes", [[512],[256],[1024],[1024,512],[1024,1024],[512,512],[512,512,512],[1024,1024,1024],[1024,512,512]]),
       ("weight_decay_penalty",[0.0001])])

params_dict=OrderedDict([
        ("batch_size", [64]),
        ("learning_rate", [0.00001,0.0001]),
        ("weight_init_stddevs",[.033]),
        ("bias_init_consts",[0.1]),
        ("dropouts",[0.5]),
        ("layer_sizes", [[3000,500,500,3000],[1024],[500,3000]]),
       ("weight_decay_penalty",[0.0001])])

def model_builder(model_params, model_dir):
    model =mk.multitask(n_tasks=len(m),
               n_features=1024, model_dir=model_dir,**model_params)




    return model

metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score,np.mean),
          dc.metrics.Metric(dc.metrics.mean_absolute_error,np.mean)]
mulu=os.path.join('/home/hdd2/lost_data/sequence_ic50_data_dnn')
if not os.path.exists(mulu):
    os.mkdir(mulu)


optimizer =HyperOPTwithEarlyStop6_4.HyperOPT_EarlyStop(model_builder,mulu,target=m)
all_results = optimizer.early_stopping(params_dict=params_dict,
                                       train = train_dataset,
                                       valid = valid_dataset,
                                       test = test_dataset,
                                       metrics=metric,
                                       output_transformers=transformers,
                                       max_epoch=500,
                                       early_stopping=True,
                                       direction=1,
                                       p_max =5,
                                       hp_valid_lst=['batch_size',
                                                     'learning_rate',
                                                      'weight_init_stddevs',
                                                       'bias_init_consts',
                                                      'dropouts',
                                                      'layer_sizes',
#
                                                      'weight_decay_penalty',
                                                      ])







