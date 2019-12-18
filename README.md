# Multitarget-drug-design
Here, we report an automated system comprising a deep recurrent neural network (RNN) 
and a multitask deep neural network (MTDNN) to design and optimize multitargeted antipsychotic drugs. 


To run the code:


##Requirements
This package requires:
Python 3.6
Deepchem
Keras
Scikit-Learn

##Usage
To establish a multitask DNN model, you need to process your data and run './multitask-dnn/multitask_dnn.py', 
you can also run ''./multitask-dnn/HyperOPTwithEarlyStop6.py'
to search your best hyperparameters.
For this study, you need to establish 3 multitask DNN model, 
where the PKi DNN model was used to transfer well-trained inital parameters to EC50/IC50
DNN models.Then you need to use './generating_rnn/training.py' to train the Prior.
A pretrained Prior is include. Then you can run './generating_rnn/main.py' to 
train models for GPCRs ligands.
