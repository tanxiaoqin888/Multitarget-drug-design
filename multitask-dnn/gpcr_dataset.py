#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:44:16 2018

@author: deepchem
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os

import deepchem
import linecache
from deepchem.data import DiskDataset
from HP import Harmonious_positive
from OnePS import OnePositiveSplit
def load_gpcr(dataset_file, featurizer='ECFP',transformers=True, reload=True, sep='OnePositiveSplit',K=5):
#    data_dir=os.path.dirname(dataset_file)
    
    save_dir=os.path.join(os.path.dirname(dataset_file), '.'.join(os.path.basename(dataset_file).split('.')[:-1]),"ecfp","split")
    train, valid, test = os.path.join(save_dir, 'train'), os.path.join(
      save_dir, 'valid'), os.path.join(save_dir, 'test')
    fopen=open(dataset_file,"r")
    ss=fopen.readlines()
    m=ss[0].strip('\n').split(',')
    m.remove('SMILES')   
    if os.path.isdir(save_dir):
        if reload:
            dataset,train_dataset, valid_dataset, test_dataset =DiskDataset(data_dir=save_dir), DiskDataset(data_dir=train), DiskDataset(data_dir=valid),DiskDataset(data_dir=test)
            transformers = [
					deepchem.trans.NormalizationTransformer(transform_w=True, dataset=train_dataset)
					]
            all_dataset = (dataset,train_dataset, valid_dataset, test_dataset)
            return m, all_dataset, transformers
    if featurizer == 'ECFP':
        featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = deepchem.feat.AdjacencyFingerprint(max_n_atoms=150, max_valence=6)
    elif featurizer == 'SelfDefine':
        featurizer = deepchem.feat.UserDefinedFeaturizer(feature_field)
    loader = deepchem.data.CSVLoader(tasks=m, smiles_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(dataset_file,data_dir=save_dir, shard_size=8192)
#    dataset = loader.featurize(dataset_file, shard_size=8192)
    # Initialize transformers
    if transformers:
        transformers = [ deepchem.trans.NormalizationTransformer(transform_w=True, dataset=dataset) ]
        for transformer in transformers:
            dataset = transformer.transform(dataset)
    splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'random_stratified': deepchem.splits.RandomStratifiedSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'task': deepchem.splits.TaskSplitter(),
      'Harmonious_positive':Harmonious_positive(),
      
      'OnePositiveSplit':OnePositiveSplit()
    }
    splitter = splitters[sep]
    if sep == 'task':
        fold_datasets = splitter.k_fold_split(dataset, K)
        all_dataset = fold_datasets
    elif sep == 'Harmonious_positive':
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
                dataset)
        train_dataset = DiskDataset.from_numpy(train_dataset.X,train_dataset.y,train_dataset.w,train_dataset.ids,dataset.tasks,data_dir=train)
        valid_dataset = DiskDataset.from_numpy(valid_dataset.X,valid_dataset.y,valid_dataset.w,valid_dataset.ids,dataset.tasks,data_dir=valid)
        test_dataset  = DiskDataset.from_numpy(test_dataset.X,test_dataset.y,test_dataset.w,test_dataset.ids,dataset.tasks,data_dir=test)
        all_dataset = (dataset,train_dataset, valid_dataset, test_dataset)
    elif sep == 'Harmonious_positive' and K:
#        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
#                                dataset,
#                                frac_train=frac_train,
#                                frac_valid=0,
#                                frac_test=1- frac_train,
#                                )
#        train_dataset = DiskDataset.from_numpy(train_dataset.X,train_dataset.y,train_dataset.w,train_dataset.ids,
#                                               dataset.tasks,data_dir=train)
#        train_dataset.reshard(8192)
#        test_dataset  = DiskDataset.from_numpy(test_dataset.X,test_dataset.y,test_dataset.w,test_dataset.ids,
#                                               dataset.tasks,data_dir=test)
#        test_dataset.reshard(8192)
#        fold_dataset = splitter.k_fold_split(
#                train_dataset, K, directories=[os.path.join(valid,str(i)) for i in range(K)],verbose=True)
        fold_dataset = splitter.k_fold_split(
                dataset, K, directories=[os.path.join(valid,str(i)) for i in range(K)],verbose=True)
        folds=[]
        for i in range(K):
            print('merge fold dataset {}...'.format(i))
            train_fold = DiskDataset.merge(
                    [fold_dataset[j] for j in range(K) if j!=i],
                    merge_dir=os.path.join(valid,str(i),'train_fold'))
            test_fold = DiskDataset.merge([fold_dataset[i]],merge_dir=os.path.join(valid,str(i),'valid_fold'))
            folds.append([train_fold,test_fold])
        all_dataset = (dataset,[], folds, [])
    else:
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
                             dataset,
                             train_dir=train, 
                             valid_dir=valid, 
                             test_dir=test,
                             frac_train=frac_train,
                             frac_valid=frac_valid,
                             frac_test=frac_test)
        all_dataset = (dataset,train_dataset, valid_dataset, test_dataset)
        

                  
                    
                    
                
        
#    else:
#        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset,train_dir=train, valid_dir=valid, test_dir=test)
#        all_dataset = (dataset,train_dataset, valid_dataset, test_dataset)
#    if reload:
#        deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,transformers)
    return m, all_dataset, transformers
