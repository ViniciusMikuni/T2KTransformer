import numpy as np
from sklearn.utils import shuffle
import sys
import os
import tensorflow as tf
import gc



class DataLoader():
    def __init__(self, path, batch_size=512,rank=0,size=1):
    
        self.path = path
        self.X = np.load(self.path)['events'][rank::size]        
        self.global_var = np.load(self.path)['global_var'][rank::size]
        self.mask = self.X[:,:,0]!=0

        self.batch_size = batch_size
        self.nevts = np.load(self.path)['events'].shape[0]
        self.num_part = self.X.shape[1]
        self.num_classes = 1

        self.num_feat = self.X.shape[2]
        self.num_global = self.global_var.shape[1]
        self.steps_per_epoch = None #will pass none, otherwise needs to add repeat to tf data
        self.files = [path]
        self.make_onehot_pid()

    def make_onehot_pid(self):
        pid = self.X[:,:,-1]
        #Will create a charge feature and split the pid as a one hot encoded
        new_X = np.zeros((self.X.shape[0],self.num_part,self.num_feat+8))
        new_X[:,:,:self.num_feat-1] = self.X[:,:,:self.num_feat-1]
        new_X[:,:,self.num_feat] = np.sign(pid) * ((pid!=14) & (pid!=111))
        new_X[:,:,self.num_feat+1] = np.abs(pid)==11
        new_X[:,:,self.num_feat+2] = np.abs(pid)==13
        new_X[:,:,self.num_feat+3] = np.abs(pid)==14
        new_X[:,:,self.num_feat+4] = np.abs(pid)==111
        new_X[:,:,self.num_feat+5] = np.abs(pid)==211
        new_X[:,:,self.num_feat+6] = np.abs(pid)==2112
        new_X[:,:,self.num_feat+7] = np.abs(pid)==2212
        

        self.X = new_X
        #Update number of features
        self.num_feat = self.X.shape[2]


    def combine(self,datasets):
        self.label = np.zeros((self.X.shape[0],1))
        for dataset in datasets:
            self.nevts += dataset.nevts
            self.X = np.concatenate([self.X,dataset.X],0)
            self.mask = np.concatenate([self.mask,dataset.mask],0)
            self.global_var = np.concatenate([self.global_var,dataset.global_var],0)
            self.label = np.concatenate([self.label,np.ones((dataset.X.shape[0],1))],0)


        self.X,self.mask,self.global_var,self.label= shuffle(self.X,self.mask,self.global_var,self.label)
        
        
    def make_tfdata(self):
        self.X = self.preprocess(self.X,self.mask).astype(np.float32)
        self.global_var = self.preprocess_event(self.global_var).astype(np.float32)

        training_data = {'input_features':self.X,
                         'input_global':self.global_var}

        
        
        return training_data,self.label


    def preprocess(self,x,mask):
        self.mean_part = [ 2.69749515e+00, -9.41835143e-07, -7.02677277e-07,  2.52753799e+00,
                           0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.std_part = [4.63607129e+00, 3.36605236e-01, 3.36497599e-01, 4.68892480e+00,
                         1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
        return mask[:,:, None]*(x-self.mean_part)/self.std_part

    def preprocess_event(self,x):
        self.mean_global =  [ 4.97560348  ,5.26390402 ,12.65448029]
        self.std_global  = [4.24227652 ,4.23591129 ,4.9318715]

        return (x-self.mean_global)/self.std_global

    def revert_preprocess(self,x,mask):                
        new_part = mask[:,:, None]*(x*self.std_part + self.mean_part)
        return  new_part

    def revert_preprocess_jet(self,x):
        new_x = self.std_global*x+self.mean_global
        return new_x
