# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 08:39:13 2021

@author: brand
"""
from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import math

def softmax(X, T, M, v, alpha):
    '''
    Args: 
        X - Training dataset with dimensions dxn
        T - Class labels 
        M - Epochs
        v - validation set 
        alpha - learning rate
        
    Returns:
        tbd
        target 60 by 4
        data 60001 by 60
        weight 60001 by 4
        length 60
        width 4
        length2 60001
        width2 60
        '''
    
    runs = 0
    
    
    length,width = np.shape(T)
    length2,width2 = np.shape(X)
    
    T = np.transpose(T)
    
    weight = np.zeros((length2,width))
    y = np.zeros((width, width2))
    
    while runs < M:
        y = np.matmul(np.transpose(weight), X)
        for i in range(width):
            for j in range(width2):
                y[i][j] = math.exp(y[i][j])
        
        coltot = y.sum(axis=0)
        
        for i in range(width):
            for j in range(width2):
                y[i][j] = y[i][j] / coltot[j]
        
        
        for p in range(length):
            for m in range(width):
                for n in range(length2):
                    weight[n][m] = weight[n][m] + alpha * (T[m][p] - y[m][p]) * X[n][p]
    
      
        runs += 1

    return y


dimension,sample = np.shape(a)
bias = np.ones((1,sample))  
a = np.vstack((bias,a))
    
dimension1,sample1 = np.shape(V)
bias = np.ones((1,sample1))  
V = np.vstack((bias,V))
    
dimension2,sample2 = np.shape(B)
bias = np.ones((1,sample2))  
B = np.vstack((bias,B))