# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:44:04 2017

@author: unmeshphadke
"""

'''
First module : Multiple Linear Regression 

'''
import pandas as pd
import numpy as np
import random

class LinearRegressor():
    '''
    Object for defining Linear Regressor in Python
    ---------
    Parameters
    ---------
    input  ,dict = {variable name_1 : [List/array] , 
                          variable_name_2 : [List/array], ....}
    output , single key dict = {variable name : [List/array]}
    
    -----
    Examples
    ------
    
    '''
    def __init__(self,input,output):
        self.input=pd.DataFrame(input)
        self.output=output
        
    def input_shape(self):
        df=self.input
        #inp= pd.DataFrame(self.input)
    
        return df.shape
        

    def predict(self,x,w):
        '''
        Returns the output of applying the weights to the input x
        ----
        Parameters:
        
        x: List or array, with first element as 1 to account for the bias term
        w: Set of weights, list or array
        '''
        return (np.dot(np.array(x),np.array(w)))
        
    def _error(self,x,y,w):
        '''
        Returns the error after applying the weight w to the input vector x
        '''
        return y - self.predict(x,w)
        
    def errorsquared(self,x,y,w):
        '''
        Returns error square
        '''
        return self._error(x,y,w)**2
        
    def gradient(self,x,y,w):
        ''' 
        Returns the gradient vector for a particular squared error term
        
        '''
        
        return np.array([-2*x_i* self._error(x,y,w) for x_i in np.array(x).tolist()])
        
    def fit(self,x,y,solver_method,**kwargs):
       '''
       Returns the set of weights with the first term in the weight vector 
       correesponding to the bias term
       
       Parameter
       ---------
       x : Input  training data
       y : Output training data
       solver_method : Learning method used to learn weights. For eg : SGD
       ---------
       '''
       w_initial = [random.random() for x_i in x[0]]
       learning_rate = kwargs.get("learning_rate",None)
       
       if learning_rate:
           return solver_method(self.errorsquared,self.gradient,x,y,w_initial,learning_rate)
       else:
           return solver_method(self.errorsquared,self.gradient,x,y,w_initial,0.001)
       
        
        
        
        
    

