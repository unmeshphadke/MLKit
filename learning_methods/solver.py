# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 13:18:30 2017

@author: unmeshphadke

Reference  : "Data Science from Scratch, Joel Grus "
"""

'''
Functions for solving for minimising loss functions

''' 

import random
import numpy as np
import pandas as pd
'''
---------------------------
STOCHASTIC GRADIENT DESCENT
---------------------------
'''

def random_order(data):
    '''
    Returns the dataset shuffled in random order
    '''
    ind = [i for i,_ in enumerate(data)]
    random.shuffle(ind)
    for i in ind:
        yield data[i]
        
        
def SGD(loss,gradient,x,y,w_0,alpha_0 = 0.01,**kwargs):
    '''
    Returns the optimum value of the weights w
    
    Parameters
    ----------
    loss :  Loss function to be minimised
    
    gradient : Expression for the gradient
    
    x: Input data

    y : Output data

    w_0: Initial guess for the weights

    alpha : Initial Learning rate. is modified if no appreciable learning is taking place
    '''
    num_iterations = kwargs.get('iterations',0)
    #If no. of iteratiosn not mentioned set it to 0. Means that iterate tillconvergence
    
    data = zip(x,y)
    w = w_0
    alpha = alpha_0
    min_w,min_error = None , float("inf")
    
    if num_iterations > 0 :
        for i in range(num_iterations):
            for x_i,y_i in random_order(data):
                grad = gradient(x_i,y_i,w)
                w = w - alpha*grad
        return w
        
    else:
        
        #This is the intelligent mode. Learning rate is dynamically modified and
        #convergence is checked too.
        #Reference : Chapter 15: Data Science from Scratch, Joel Grus
        
        #If there is no learning taking place for100 iterations, stop
        it_no_learning = 0 #Iterations with no larning set to 0
        
        while it_no_learning < 100 :
            
            error = sum(loss(x_i,y_i,w) for x_i,y_i in data)
            
            if error < min_error :
                min_error = error
                min_w = w
                it_no_learning = 0
                alpha= alpha_0 #et learning rate to specified one
            else:
                it_no_learning = it_no_learning + 1
                alpha = alpha * 0.9
            
            for x_i,y_i in random_order(data):
                grad = gradient(x_i,y_i,w)
                w = w - alpha*grad
            
        return min_w   
                
            
        
    
                
                
            