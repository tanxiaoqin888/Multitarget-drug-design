#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:53:29 2018

@author: txqq
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import os
import csv
import itertools
import tempfile
import deepchem as dc
import collections
import pandas as pd
from deepchem.utils.evaluate import Evaluator
from deepchem.utils.save import log
import tensorflow as tf
"""
  Provides simple hyperparameter grid search and early stopping capabilities.
"""
class HyperOPT_EarlyStop(object):
    
    def __init__(self, 
                 model_class, 
                  mulu,
                 verbose = True,
                 target=0): #chushihua
        self.model_class = model_class
        self.verbose = verbose
        self.logdir = mulu
        self.target=target
    
        
    def early_stopping(self,
                    params_dict,
                    train,
                    valid,
                    test,
                    metrics,
                    output_transformers=[],
                    max_epoch=1600,
                    early_stopping=True,
                    direction=1,
                    p_max =4,
                    hp_valid_lst=['batch_size',
                                'learning_rate',
                                'weight_init_stddevs',
                                'bias_init_consts',
                                'dropouts',
                                'layer_sizes',
#                                                      
                                'weight_decay_penalty',
                                  ]
                                ):
        """
        params_dict : for example
            params_dict = {
               "batch_size": [i for i in range(32,64+1,16)],
               "learning_rate": [1e-3,3e-4],
               "dropouts": [0.5],
               "layer_sizes":[[i,j] for i,j in zip(
                               range(500,1000+1,200),range(600,1500+1,200))],
               "weight_decay_penalty": [0.001,0.002],
               "weight_decay_penalty_type":['l2']
              }
        metrics : the first metric in metrics is used in early stopping
        direction :when you want bigger metrics[0], direction should be 1.
                    when you want bigger metrics[0], direction should be -1.
        p_max : patience for bad valid set proformance
        early_stopping : if early_stopping is false,then run until epoch=max_epoch           
        hp_valid_lst : you can add the hyperparams which you want to 
                    search and show in your csv file into hp_valid_lst.
        """
        hyperparams = params_dict.keys()
        hyperparam_vals = params_dict.values()
        
        for hyperparam_list in params_dict.values():
            assert isinstance(hyperparam_list, collections.Iterable)
       
        for ind, hyperparameter_tuple in enumerate(itertools.product(*hyperparam_vals)):
              model_params=collections.OrderedDict()
    #          
              for hyperparam, hyperparam_val in zip(hyperparams, hyperparameter_tuple):
                      model_params[hyperparam] = hyperparam_val
    #                   
              log("ind: %s" % str(ind), self.verbose)
              log("hyperparameters: %s" % str(model_params), self.verbose)
              
              output_file = os.path.join(self.logdir,'HyperOpt_EarlyStop_633.csv')#jia/
              
              self._write_title(ind,model_params,hp_valid_lst,output_file)
              
             
              model_dir = self._model_dir_maker(self.logdir, ind, hyperparameter_tuple)
              
              model = self.model_class(model_params, model_dir)
              gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
              config=tf.ConfigProto(gpu_options=gpu_options)
              
              p = 0 ##patience for bad valid set proformance
              valid_scores_for_epochs =[]
              for epoch in range(1,max_epoch+1+p_max):
                  
                  model.fit(train,
                            nb_epoch=1,
#                            restore= epoch>1,
                            model_dir = model_dir,
                            model_name = 'epoch_'+str(epoch-1)+'.pickle',
                            **model_params,
                            configproto=config,
                            )                  
#                  model.save(model_dir,model_name = 'epoch_'+str(epoch)+'.pickle')
                  
                  
                  scores =self._evaluate(model,metrics,output_transformers,train,valid,test)
                  metri=[dc.metrics.Metric(dc.metrics.mean_absolute_error,mode="regression"),dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]
                  valscore=model.evaluate(test,metri,output_transformers)
                  key=[]
                  for s2 in valscore.keys():
                      key.append(str(s2))
                  model_test=pd.DataFrame(valscore,columns=key,index=self.target)
#                  dopamine=model_test.loc[["P14416","P21728","P08908","P28223"]]
#                  dopamine=model_test.loc[["P14416","P21728","P08908","P28223"]]
                  if not os.path.exists(model_dir+'/epoch_'+str(epoch)):
                      os.makedirs(model_dir+'/epoch_'+str(epoch))
                  model_test.to_csv(model_dir+'/epoch_'+str(epoch)+'/quan_yanzheng'+'.csv')
#                  dopamine.to_csv(model_dir+'/epoch_'+str(epoch)+'/dopamine_yanzheng'+'.csv')
                  
                  if(scores[1][[i.name for i in metrics][0]]>0.99):
                      model.save(model_dir,model_name = 'epoch_'+str(epoch)+'.pickle')
                  
                  if epoch <= max_epoch:
                      self._write_score(epoch,scores,output_file)
                  
                  valid_scores_for_epochs.append(
                                  scores[1][[i.name for i in metrics][0]])
                  if early_stopping and epoch > 1:
                      if direction*valid_scores_for_epochs[-1] < direction*valid_scores_for_epochs[-2]:
                          p +=1
                      else:
                          p = 0
                  if epoch == max_epoch+p_max or (p == p_max and scores[1][[i.name for i in metrics][0]]>0.85):
                      model.save(model_dir,model_name = 'epoch_'+str(epoch)+'.pickle')
                      model.save(model_dir,model_name = 'epoch_'+str(epoch-p_max)+'.pickle')
                      best_epoch_model = self.model_class(model_params, model_dir)
                      best_epoch_model.load_from_dir(model_dir,'epoch_'+str(epoch-p_max)+'.pickle')
                      test_score = self._evaluate(best_epoch_model,metrics,output_transformers,test)
                      metri = [dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")]
                      testscore = best_epoch_model.evaluate(test,metri,output_transformers)
                      print("it is scors")
                      print(testscore)
                      print("hahaha")
                      print(test_score)
                      key=[]
                      for s3 in testscore.keys():
                          key.append(str(s3))
                      model_test=pd.DataFrame(testscore,columns=key,index=self.target)
#                      dopamine=model_test.loc[["P14416","P21728","P08908","P28223"]]
#                      dopamine=model_test.loc[["P14416","P21728","P08908","P28223"]]
                      if not os.path.exists(model_dir+'/epoch_'+str(epoch-p_max)):
                          os.makedirs(model_dir+'/epoch_'+str(epoch-p_max))
                      model_test.to_csv(model_dir+'epoch_'+str(epoch-p_max)+'quan_test'+'.csv')
#                      dopamine.to_csv(model_dir+'epoch_'+str(epoch-p_max)+'dopamine_test'+'.csv')
                      titlelst=['epoch']
                      for key in test_score[0].keys():
                          titlelst.append('test'+key)
                      rowlst=[epoch-p_max]
                      for value in test_score[0].values():
                          rowlst.append(value)
                      with open(os.path.join(output_file),'a') as csvfile:
                          f = csv.writer(csvfile)
                          f.writerow(titlelst)
                          f.writerow(rowlst) 
                      break
           
                    
    def _model_dir_maker(self,logdir,ind,hyperparameter_tuple):
        if logdir is not None:
            model_dir = os.path.join(
                logdir, 
                'nb'+str(ind)+'_'+str(hyperparameter_tuple).replace(',','-').replace('[','(').replace(']',')'))
            log("model_dir is %s" % model_dir, self.verbose)
            try:
                os.makedirs(model_dir)
            except OSError:
              if not os.path.isdir(model_dir):
                log("Error creating model_dir, using tempfile directory",
                    self.verbose)
                model_dir = tempfile.mkdtemp()
        else:
            model_dir = tempfile.mkdtemp()
        return model_dir
    def _evaluate(self,model,metrics,output_transformers,*datasets):
        scores=[]
        for dataset in datasets: 
            evaluator= Evaluator(
                          model, dataset, output_transformers)
            score = evaluator.compute_model_performance(metrics)
            scores.append(score)
#            print(scores)
        return scores
    def _write_title(self,ind,model_params,hp_valid_lst,output_file):
        paramskeys=[i for i in hp_valid_lst]
        paramsvalues=[model_params[i]  for i in hp_valid_lst]
        
              
        with open(os.path.join(output_file),'a') as csvfile:
                  f = csv.writer(csvfile)
                  f.writerow([str(ind)])
                  f.writerow(paramskeys)
                  f.writerow(paramsvalues)
                  
    def _write_score(self,epoch,scores,output_file):
        if epoch ==1:
            titlelst=['epoch']
            for key in scores[0].keys():
                  titlelst.append('train_'+key)
                  titlelst.append('valid_'+key)
                  titlelst.append('test_'+key)
        rowlst=[epoch]
        for trainvalue,validvalue,testvalue in zip(scores[0].values(),scores[1].values(),scores[2].values()):
              rowlst.append(trainvalue)
              rowlst.append(validvalue)
              rowlst.append(testvalue)    
        with open(os.path.join(output_file),'a') as csvfile:
              f = csv.writer(csvfile)
              if epoch == 1:
                  f.writerow(titlelst)
              f.writerow(rowlst)      
 
