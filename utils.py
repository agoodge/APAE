# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:04:12 2019

@author: Adam
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import sklearn.metrics as skm
from math import e
from model import train_model

import variables as var

###################################### data fns #############################################

def import_data(dataset): 
    # redacted
    pass

# arranges normal data into a sliding window of length seq_len
def split_data(sequence, seq_len):
    
    X = []
    for i in range(len(sequence)):
        end_idx = i + seq_len
        # check if reached the end of sequence
        if end_idx > len(sequence):
            break
        seq_x = sequence[i:end_idx]
        X.append(seq_x)
        
    return np.array(X)

#create dataloaders of (x,y) data and labels
def create_train_set(normal_x):

    #split into train and validation sets
    val_size = int(0.15*len(normal_x))
    train_dataset, val_dataset = random_split(normal_x, [val_size,len(normal_x)-val_size])

    #create dataloaders for iteration
    train_loader = DataLoader(dataset=train_dataset, batch_size = var.batch_size, shuffle = True)
    val_loader = DataLoader(dataset=val_dataset, batch_size= var.batch_size, shuffle = True)
    
    return train_loader, val_loader


########################### other fns ###################################

def threshold_errors(errors, percentile, error_path):
    #finding the reconstruction error for each test example from attack data
    real_err = torch.load(error_path)
    if percentile < 100:
        threshold = np.percentile(real_err, percentile)
    elif percentile >= 100:
            threshold = (percentile/100)*max(real_err)
    predicted_labels = np.zeros_like(errors)
    for i in range(len(errors)):
        if errors[i] > threshold:
            predicted_labels[i] = 1
    return predicted_labels

def prob_errors(errors,percentile,error_path):
    real_err = torch.load(error_path)
    if percentile < 100:
        threshold = np.percentile(real_err, percentile)
    elif percentile >= 100:
            threshold = (percentile/100)*max(real_err)
    predicted_labels = 1/(1+e**(1-errors/threshold))
    return predicted_labels


#function for plotting regions of labels
def heads_tails(labels):
    heads = []
    tails = []
    for i in range(len(labels)-1):
        if labels[i] == 1:
            if labels[i-1] == 0:
                heads.append(i)
            if labels[i+1] == 0:
                tails.append(i)
    return heads, tails
    
#score metrics
def metrics(ytrue, ypred):
    ypred = (ypred > 0).astype(int)
    return skm.precision_score(ytrue,ypred), skm.recall_score(ytrue,ypred), skm.f1_score(ytrue,ypred)