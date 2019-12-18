#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:15:05 2018

@author: txqq
"""

from evaluate import evaluate_smiles as es
import os
def filter_generate_smiles(number,valid_smiles,path):
        
    
    valid_smiles1=es().filter_chain_too_long(valid_smiles)
    
    valid_smiles2=es().filtered_sa(valid_smiles1)
    
    valid_smiles3=es().filter_druglikeness_5_rules(valid_smiles)
    
    valid_smiles4=es().filtered_qed(valid_smiles3)
    
    filtered_smiles=valid_smiles4
    filtered_smiles_num=len(filtered_smiles)
    print("Last_remain_smiles:%i"%filtered_smiles_num)
#    path='/home/txqq/work/RNN/lstm5/filter_smiles/data/copy_guiyihua_all_lstm'
    if not os.path.exists(path):
        os.mkdir(path)
    lstmpath=os.path.join(path,'filter_5rules_smiles'+str(number)+'.txt')
    
    f3=open(lstmpath,'w')
    for i in filtered_smiles:
        f3.write(i+'\n')
    f3.close()
    
    return filtered_smiles
