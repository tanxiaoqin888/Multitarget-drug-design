#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:40:15 2018

@author: deepchem
"""
from collections import OrderedDict
import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed
from sascorer import calculateScore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.Descriptors import qed, MolLogP
import math
from rdkit.Chem import MolSurf, Crippen
from rdkit.Chem import rdMolDescriptors as rdmd


class evaluate_smiles():
    
  
    def valid_smi(self,smile):
        if smile=="":
            judge=False
        else:
            
            mol=Chem.MolFromSmiles(smile,sanitize=True)
            if mol is not None:
                judge=True
            else:
                judge=False
        return judge
    
    def valid_smiles_filter(self,smiles):#smiles are a list of smile
        original_smiles_num=len(smiles)
        count=0
        for smi in smiles:
            if self.valid_smi(smi)==False:
               smiles.remove(smi)
               count=count+1
        valid_smiles_num=len(smiles)
        valid_rate=valid_smiles_num/original_smiles_num
        valid_rate=str(valid_rate)
        print("original_smiles:%i"%original_smiles_num)
        print("valid_smiles_rate:%s"%valid_rate)
        print("unavaliable_smiles:%i"%count)
        return smiles
    
    def valid_smiles(self,smiles):
        
        valid_smiles_list=[]
        for smi in smiles:
            if self.valid_smi(smi)==True:
               
                valid_smiles_list.append(smi)
        sum_valid_smiles=len(valid_smiles_list)
        valid_smiles_list_1=list(set(valid_smiles_list))
        sum_nodup_smiles=len(valid_smiles_list_1)
        valid_rate=sum_valid_smiles/len(smiles)
        no_dup_valid_rate=sum_nodup_smiles/len(smiles)
        valid_rate=str(valid_rate)
        nodup_valid_rate=str(no_dup_valid_rate)
        
        return valid_rate,nodup_valid_rate,sum_nodup_smiles,len(smiles),valid_smiles_list_1
    
    def c_chain_length(self,valid_smi):
        mol=Chem.MolFromSmiles(valid_smi)
        c=[]
        z=0
        for atom in mol.GetAtoms():
            s=str(atom.GetHybridization())
        
            if s=="SP3" and atom.GetAtomicNum()==6 :
                z+=1
               
       
        if z==0:
            return z
        elif z==1:
            return z
        else:
            
            for bond in mol.GetBonds():
                atom_begin=bond.GetBeginAtom()
                
                atom_end=bond.GetEndAtom()
                if atom_begin.GetAtomicNum()==6 and  atom_end.GetAtomicNum()==6:
                    a1=atom_begin.GetIdx()
                    a2=atom_end.GetIdx()
                    if not atom_begin.GetIsAromatic():
                        if not atom_end.GetIsAromatic():
                            if not atom_begin.IsInRing():
                                if not atom_end.IsInRing():
                                    c.append([a1,a2])
            d=c[:]
            
            for i in range(len(c)):
                for j in range(1,len(c)):
                    if c[i][-1]==c[j][0]:
                        d[i].append(c[j][1])
            
            if d==[]:
                c_l=0
            else:
                c_l=max([len(i) for i in d])
                
                
            return c_l
    def filter_chain_too_long(self,valid_smiles):
        count=0
        for i in  valid_smiles:
            if self.c_chain_length(i)>=8:
                count=count+1
                valid_smiles.remove(i)
                
        print("smiles_with_chain_too_long:%i"%count)
        return valid_smiles
    
    def filtered_sa(self,valid_smiles):
        count=0 
        for i  in valid_smiles:
            mol=Chem.MolFromSmiles(i)
            SA_score= calculateScore(mol)
            if SA_score > 5:
                valid_smiles.remove(i)
                count=count+1
        print("unavaliable_SA_mol:%i"%count)
        
        return valid_smiles
    
    def filter_druglikeness_5_rules(self,smiles):
       
        count=0
        for i in smiles:
            mol=Chem.MolFromSmiles(i) 
            mol = Chem.RemoveHs(mol)
              
            MW=rdmd._CalcMolWt(mol)
            ALOGP=Crippen.MolLogP(mol)
            HBA=rdmd.CalcNumHBA(mol)
            HBD=rdmd.CalcNumHBD(mol)
            PSA=MolSurf.TPSA(mol)
            ROTB=rdmd.CalcNumRotatableBonds(mol, rdmd.NumRotatableBondsOptions.Strict)
            
            if MW >600 or ALOGP> 6 or ALOGP<0 or HBA>11 or HBD>7 or PSA>180 or ROTB>11 :
                smiles.remove(i)
                count=count+1
        print("unavaliable rule_5_drug:%i"%count)
        
        return smiles
   
    def filtered_sa(self,valid_smiles):
        count=0 
        for i  in valid_smiles:
            mol=Chem.MolFromSmiles(i)
            SA_score= calculateScore(mol)
            if SA_score > 5:
                valid_smiles.remove(i)
                count=count+1
        print("unavaliable_SA_mol:%i"%count)
               
        return valid_smiles
    def filtered_qed(self,valid_smiles):
        count=0
        for i  in valid_smiles:
            mol=Chem.MolFromSmiles(i)
            qed_score= qed(mol)
            if qed_score < 0.4:
                valid_smiles.remove(i)
                count=count+1
        print("unavaliable QED mol:%i"%count)
        return valid_smiles
    
    
    def pair_similiar(self,valid_smiles):
        if len(valid_smiles)<=2:
            return 0,0
        else:
            
            valid_mols=[Chem.MolFromSmiles(i) for i in valid_smiles]
            valid_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024,useChirality=True) for mol in valid_mols]
            pair_similiar=[]
            for i in range(len(valid_fps)):
                for j in range(i+1,len(valid_fps)):
                    fp_i=valid_fps[i]
                    fp_j=valid_fps[j]
                    pair_similiar.append(DataStructs.TanimotoSimilarity(fp_i,fp_j))
            
            pair_similiar_numpy=np.array(pair_similiar)
            very_similiar=pair_similiar_numpy[pair_similiar_numpy>0.75]
            very_similiar_rate= very_similiar.shape[0]/len(pair_similiar)
            mean_pair_similiar=sum(pair_similiar)/len(pair_similiar)
            return str(very_similiar_rate),str(mean_pair_similiar)
    
    
    def length_of_chian_distirbution(self,valid_smiles):
        dict1=OrderedDict()
        dict2=OrderedDict()
        count_valid_smiles=len(valid_smiles)
        if count_valid_smiles <= 5:
            dict2[">7"]='0'
        else:
            for i in valid_smiles:
                carbonchain_length=self.c_chain_length(i)
                carbonchain_length=str(carbonchain_length)
                try:
                    dict1[carbonchain_length]+=1
                except:
                    dict1[carbonchain_length]=1
            a=[0,1,2,3,4,5,6,7]
            b=[str(i) for i in a]
       
           
            dict2=OrderedDict()
            h=0
            for k  in b:
                try:
                    dict2[k]=dict1[k]/count_valid_smiles
                    h+=dict1[k]
                except:
                    dict2[k]=0
            res=count_valid_smiles-h
            dict2[">7"]=res/count_valid_smiles
        return dict2
    
     

    def qed_evaluate(self,valid_smiles):
         qed_lst=[]
         for i in valid_smiles:
             mol=Chem.MolFromSmiles(i)
             qed_score=qed(mol)
             qed_lst.append(qed_score)
         return qed_lst
    def SA_evaluate(self,valid_smiles):
        SA_lst=[]
        for i in valid_smiles:
            mol=Chem.MolFromSmiles(i)
            SA_score= calculateScore(mol)
            SA_lst.append(SA_score)
        return SA_lst  
    def pair_similiar_fcfp4(self,valid_smiles):
        if len(valid_smiles)<2:
            return 0,0
        else:
            
            valid_mols=[Chem.MolFromSmiles(i) for i in valid_smiles]
            valid_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024,useFeatures=True) for mol in valid_mols]
            pair_similiar=[]
            for i in range(len(valid_fps)):
                for j in range(i+1,len(valid_fps)):
                    fp_i=valid_fps[i]
                    fp_j=valid_fps[j]
                    pair_similiar.append(DataStructs.FingerprintSimilarity(fp_i,fp_j))
            
            pair_similiar_numpy=np.array(pair_similiar)
            very_similiar=pair_similiar_numpy[pair_similiar_numpy>0.75]
            very_similiar_rate= very_similiar.shape[0]/len(pair_similiar)
            mean_pair_similiar=sum(pair_similiar)/len(pair_similiar)
            return str(very_similiar_rate),str(mean_pair_similiar)
  
    def pair_similiar_filter(self,valid_smiles,control_smiles):
        if len(valid_smiles)<=2:
            return 0,0
        else:
            
            valid_mols=[Chem.MolFromSmiles(i) for i in valid_smiles]
            valid_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024,useChirality=True) for mol in valid_mols]
            pair_similiar_control=[]
            control_mols=[Chem.MolFromSmiles(i) for i in control_smiles]
            control_fps = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=1024,useChirality=True) for mol in control_mols]
            
            for i in range(len(valid_fps)):
                for j in range(len(control_fps)):
                    fp_i=valid_fps[i]
                    fp_j=control_fps[j]
                    pair_similiar_control.append(DataStructs.TanimotoSimilarity(fp_i,fp_j))
            
            pair_similiar_control_numpy=np.array(pair_similiar_control)
            very_similiar_control_8=pair_similiar_control_numpy[pair_similiar_control_numpy>0.8]
            very_similiar_control_6=pair_similiar_control_numpy[pair_similiar_control_numpy>0.6]
            no_similiar_control_5=pair_similiar_control_numpy[pair_similiar_control_numpy<0.5]
            very_similiar_rate_8= very_similiar_control_8.shape[0]/len(pair_similiar_control)
            very_similiar_rate_6= very_similiar_control_6.shape[0]/len(pair_similiar_control)
            no_similiar_rate_5= no_similiar_control_5.shape[0]/len(pair_similiar_control)
            mean_pair_similiar=sum(pair_similiar_control)/len(pair_similiar_control)
            return str(very_similiar_rate_8),str(mean_pair_similiar)
                
                         
            
            
    
if __name__=="__main__":
    a=["CCCCCCCC","CC"]
    
  
