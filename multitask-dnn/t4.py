#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:39:37 2018

@author: txqq
"""
import deepchem
import numpy as np
import os
import json
import time
from deepchem.data import DiskDataset
class  target_4_save():
    def save_metadata(self,tasks, metadata_df, data_dir):

      if isinstance(tasks, np.ndarray):
        tasks = tasks.tolist()
      metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
      tasks_filename = os.path.join(data_dir, "tasks.json")
      with open(tasks_filename, 'w') as fout:
        json.dump(tasks, fout)
      metadata_df.to_csv(metadata_filename, index=False, compression='gzip')
    def shard_generator(self,compound_index,target_4_index,dataset):
        i=0
        shard=8192
        for i in range(len(compound_index)//shard+1):
            up=i*shard
          
            down=min((i+1)*shard,len(compound_index))
            print('down')
            print(down)
            y=dataset.y[up:down,target_4_index]
            print(y.shape)
            X=dataset.X[up:down]
            w=dataset.w[up:down,target_4_index]
            ids=dataset.ids[up:down]
            yield X, y, w, ids
            
    def target_4_dataset_save(self,dataset,file):
        compound=dataset.ids.tolist()
        target=dataset.get_task_names()
        print(target)
        w=dataset.w
        print('w.shape')
        print(w.shape)
        compuond_4_target=[]
        target_4=['P21728','P14416','P08908','P28223']
        
        target_4=sorted(target_4,key=lambda x:target.index(x))
        target_4_index=[target.index(i) for i in target_4]
        print('target_4')
        print(target_4_index)
        for i in range(len(compound)):
            z=0
            for j in target_4_index:
            
                if w[i,j]>0:
                    z=z+1
            if z>0:
                compuond_4_target.append(i)
                
            
        compound_shard=[]
        
            
        dataset1=dataset.select(compuond_4_target)
        print(compuond_4_target)
        cpd=compuond_4_target
        metadata_rows=[]
        shard_generator=self.shard_generator(cpd,target_4_index,dataset1)
        for shard_num, (X, y, w, ids) in enumerate(shard_generator):
          basename = "shard-%d" % shard_num
          metadata_rows.append(
              DiskDataset.write_data_to_disk(file, basename,target_4 , X, y, w,
                                             ids))
          metadata_df = DiskDataset._construct_metadata(metadata_rows)
          self.save_metadata(target_4, metadata_df, file)
          time2 = time.time()
        