#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:05:02 2018

@author: txqq
"""

import numpy as np

from deepchem.splits import Splitter
from deepchem.data import DiskDataset
import numpy as np
from deepchem.splits import Splitter
from deepchem.data import DiskDataset

class Harmonious_positive(Splitter):
    
    def split(self,dataset,
              seed=777,
              frac_train=.8,     
              frac_valid=.1, 
              frac_test=.1,
              log_every_n=None
                              ):
        
        target_lst= dataset.get_task_names()
        cpd_lst= dataset.ids.tolist()
        print(cpd_lst)
        y_lst = dataset.y
        w_lst = dataset.w
        
        cpd_set_for_train = self._get_positive(
                                            target_lst,cpd_lst,y_lst,w_lst)
        
        r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train)
        cpd_set_for_train = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_train
        r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train)
        cpd_set_for_train = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_train
        cpd_index_for_train =[cpd_lst.index(cpd) for cpd in cpd_set_for_train]
        if frac_test == 0:
            cpd_set_for_test = set()
            cpd_index_for_test=[]
        else:
            remain_dataset = self._get_remain_dataset(
                                            cpd_lst,y_lst,w_lst,cpd_set_for_train)
            cpd_set_for_test = self._get_positive(target_lst,remain_dataset[0],
                remain_dataset[1],remain_dataset[2])
            r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train| cpd_set_for_test)
            cpd_set_for_test = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_test
            r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train| cpd_set_for_test)
            cpd_set_for_test = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_test
            cpd_index_for_test = [
                        cpd_lst.index(cpd) for cpd in cpd_set_for_test]
        
        if frac_valid == 0:
            cpd_set_for_valid = set()
            cpd_index_for_valid=[]
        else:
            remain_dataset2 = self._get_remain_dataset(
                cpd_lst,y_lst,w_lst,cpd_set_for_train|cpd_set_for_test)
           
            cpd_set_for_valid = self._get_positive(target_lst,remain_dataset2[0],
                remain_dataset2[1],remain_dataset2[2])
            r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train|cpd_set_for_test|cpd_set_for_valid)
            cpd_set_for_valid = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_valid
            r=self._get_remain_dataset(cpd_lst,y_lst,w_lst,cpd_set_for_train|cpd_set_for_test|cpd_set_for_valid)
            cpd_set_for_valid = self._get_positive(target_lst,r[0],r[1],r[2])|cpd_set_for_valid
            cpd_index_for_valid = [
                        cpd_lst.index(cpd) for cpd in cpd_set_for_valid]
            
        
        assert len(cpd_set_for_train & cpd_set_for_test)==0
        assert len(cpd_set_for_train & cpd_set_for_valid)==0
        assert len(cpd_set_for_valid & cpd_set_for_test)==0
        assert len(set(cpd_index_for_train) & set(cpd_index_for_test))==0
        assert len(set(cpd_index_for_train) & set(cpd_index_for_valid))==0
        assert len(set(cpd_index_for_valid) & set(cpd_index_for_test))==0
        #start to split remain cpds randomly
        train_ratio,test_ratio  =self._ratio_remain_samples_to_spilt(
                                        cpd_lst,
                                        cpd_set_for_train,
                                        cpd_set_for_valid,
                                        cpd_set_for_test,                            
                                        frac_train,
                                        frac_valid,
                                        frac_test
                                        )
        
        cpd_index_to_randomsplit = list(set(
            range(len(cpd_lst)))-set(
            cpd_index_for_train)-set(cpd_index_for_test)-set(cpd_index_for_valid))
        train_cutoff = int(train_ratio*len(cpd_index_to_randomsplit))
        test_cutoff =  int((train_ratio+test_ratio)*len(cpd_index_to_randomsplit))
        if not seed is None:
            np.random.seed(seed)
        cpd_index_to_randomsplit_shuffled= np.random.permutation(
                                                    cpd_index_to_randomsplit)
        cpd_index_train_all = set(cpd_index_to_randomsplit_shuffled[
                :train_cutoff])|set(cpd_index_for_train)
            
        cpd_index_test_all = set(cpd_index_to_randomsplit_shuffled[
                train_cutoff:test_cutoff])|set(cpd_index_for_test)
            
        cpd_index_valid_all = set(cpd_index_to_randomsplit_shuffled[
                test_cutoff:])|set(cpd_index_for_valid)
        
        assert len(cpd_index_train_all & cpd_index_test_all)==0
        assert len(cpd_index_train_all & cpd_index_valid_all)==0
        assert len(cpd_index_valid_all & cpd_index_test_all)==0
        assert (cpd_index_train_all |cpd_index_test_all|cpd_index_valid_all) == set(range(len(cpd_lst)))
        train=list(cpd_index_train_all)
        valid=list(cpd_index_valid_all)
        test=list(cpd_index_test_all)
        return  train,valid,test
    def _get_positive(self,target_lst,cpd_lst,y_lst,w_lst):
        cpd_dict={}
        target_dict={}
        cpd_target_b={}
        for i in range(len(cpd_lst)):
            for j in range(len(target_lst)):
                if w_lst[i,j]==1:
                    cpd_target_b[(cpd_lst[i],target_lst[j])]=True
                else:
                    cpd_target_b[(cpd_lst[i],target_lst[j])]=False
       
        w0=np.sum(w_lst,axis=0)
        w1=np.sum(w_lst,axis=1)
    
        for cpd_index in range(len(cpd_lst)):
            cpd_dict[cpd_lst[cpd_index]]=w1[cpd_index]
        for tar_index in range(len(target_lst)):
            target_dict[target_lst[tar_index]]=w0[tar_index]
        cpd_set_for_one_positive_tar=set()
        one_positive_tar_lst = list()
        one_positive_cpd_lst = list()
        for target in target_dict.keys():
            if target_dict[target]==1:
                tar_column=w_lst[:,target_lst.index(target)]
                cpd_index=np.where(tar_column==1)[0][0]
           
                one_positive_cpd_lst.append(cpd_lst[cpd_index])
                one_positive_tar_lst.append(target)
        one_positive_cpd_set = set(one_positive_cpd_lst)
        cpd_set_for_one_positive_tar = one_positive_cpd_set.copy()
       
        remain_tar_set = set(target_lst) - set(one_positive_tar_lst)
        remain_tar_set=sorted(remain_tar_set,key=lambda x:target_dict[x])
        
        
        for remain_tar in remain_tar_set:
           
            b=[ cpd_target_b[cpd,remain_tar] for cpd in  cpd_set_for_one_positive_tar]
           
            if any(b):
                continue
            else:
                remain_tar_index=target_lst.index(remain_tar)
                w_remain_tar=w_lst[:, remain_tar_index]
                w_remain_positive=w_remain_tar>0
                cpd_remain_index=[]
                for i in range(len(cpd_lst)):
                    if  w_remain_positive[i]:
                        cpd_remain_index.append(i)
                
                jishuqi=[w1[i] if i in cpd_remain_index else 100  for i in range(len(cpd_lst)) ]
               
               
                w1_label=np.argmin(jishuqi)
                
              
                
                try:
                    cpd_set_for_one_positive_tar.add(cpd_lst[w1_label])
                    
                except IndexError:
                    continue   
                
                    
        return cpd_set_for_one_positive_tar    
                
    def _ratio_remain_samples_to_spilt(self,
                                        cpd_lst,
                                        cpd_set_for_train,
                                        cpd_set_for_valid,
                                        cpd_set_for_test,                            
                                        frac_train,
                                        frac_valid,
                                        frac_test
                                        ):
        
        nb_all_cpd =len(set(cpd_lst))
        nb_remain_cpd=len(
                set(cpd_lst)-set(cpd_set_for_train)-set(cpd_set_for_test)-set(cpd_set_for_valid))
        nb_train_cpd = len(cpd_set_for_train)
        nb_test_cpd = len(cpd_set_for_test)
        nb_valid_cpd = len(cpd_set_for_valid)
        if nb_all_cpd*frac_train <= nb_train_cpd and nb_all_cpd*frac_test >= nb_test_cpd and nb_all_cpd*frac_valid >= nb_valid_cpd:
            print("TRUE1")
            
            train_ratio = 0
            test_ratio = (frac_test*nb_remain_cpd+nb_valid_cpd*frac_test-nb_test_cpd*frac_valid)/(
                    (frac_test+frac_valid)*nb_remain_cpd)
        elif nb_all_cpd*frac_test <= nb_test_cpd and nb_all_cpd*frac_train >= nb_train_cpd and nb_all_cpd*frac_valid >= nb_valid_cpd:
            print("TRUE2")
            test_ratio = 0
            train_ratio = (frac_train*nb_remain_cpd+nb_valid_cpd*frac_train-nb_train_cpd*frac_valid)/(
                    (frac_train+frac_valid)*nb_remain_cpd)
        elif nb_all_cpd*frac_valid <= nb_valid_cpd and nb_all_cpd*frac_test >= nb_test_cpd and nb_all_cpd*frac_train >= nb_train_cpd:
            print("TRUE3")
            train_ratio = (frac_train*nb_remain_cpd+nb_test_cpd*frac_train-nb_train_cpd*frac_test)/(
                    (frac_train+frac_test)*nb_remain_cpd)
            test_ratio = 1-train_ratio
        elif nb_all_cpd*frac_valid <= nb_valid_cpd and nb_all_cpd*frac_test <= nb_test_cpd and nb_all_cpd*frac_train >= nb_train_cpd:
            print("TRUE4")
            train_ratio = 1
            test_ratio = 0
        elif nb_all_cpd*frac_valid <= nb_valid_cpd and nb_all_cpd*frac_test >= nb_test_cpd and nb_all_cpd*frac_train <= nb_train_cpd:
            print("TRUE5")
            train_ratio = 0
            test_ratio = 1
        elif nb_all_cpd*frac_valid >= nb_valid_cpd and nb_all_cpd*frac_test <= nb_test_cpd and nb_all_cpd*frac_train <= nb_train_cpd:
            print("TRUE6")
            train_ratio = 0
            test_ratio = 0
        elif nb_all_cpd*frac_valid < nb_valid_cpd and nb_all_cpd*frac_test < nb_test_cpd and nb_all_cpd*frac_train < nb_train_cpd:
            print("TRUE7")
            raise Exception('Number of samples are insufficient to split in that way.')
        else:    
            print("TRUE8")
            train_ratio =(-(frac_test+frac_valid)*nb_train_cpd + frac_train*nb_test_cpd +frac_train*nb_valid_cpd + frac_train*nb_remain_cpd)/(
                    (frac_train +frac_test+frac_valid)*nb_remain_cpd)
            _test_ratio =(-frac_valid*nb_test_cpd + frac_test*nb_valid_cpd + frac_test*nb_remain_cpd)/(
                    (frac_test +frac_valid)*nb_remain_cpd)
            test_ratio=(1-train_ratio)* _test_ratio
        print(str(nb_train_cpd),nb_test_cpd,nb_valid_cpd)
        print(train_ratio,test_ratio)
        return train_ratio,test_ratio
        
    def _get_remain_dataset(self,cpd_lst,y_lst,w_lst,cpd_set_for_got):
        cpd_index_for_got =[
                    cpd_lst.index(cpd) for cpd in cpd_set_for_got]
        
        remain_cpd_index = set(range(len(cpd_lst)))-set(cpd_index_for_got)
        remain_cpd_lst=[cpd_lst[i] for i in remain_cpd_index]
        remain_y_lst=np.array([y_lst[i] for i in remain_cpd_index])
        remain_w_lst=np.array([w_lst[i] for i in remain_cpd_index])
        return (remain_cpd_lst,remain_y_lst,remain_w_lst)
    
