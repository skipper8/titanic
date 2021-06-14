import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
import warnings



class titanic:
    def __init__(self, X, Y,CompX, CompY, num_it, lr, W, b, act):
        self.X = X
        self.Y = Y
        self.XC = CompX
        self.YC = CompY
        self.num_it = num_it
        self.lr = lr
        self.W = W
        print(W)
        self.b = b
        self.act = act
    
    #cost function
    def grad_cost_log_reg(self, w,b, X,Y,lambd):
        if(self.act == "sigmoid"):
            A=self.sigmoid(np.dot(X,w)+b)
            print(A)
        else:
            A = self.tanh(np.dot(X,w) + b)
        cases = int(np.size(X, 0))
        
        cost = -1/cases * np.sum(np.multiply(Y,self.safe_ln(A))+ np.multiply((1-Y),self.safe_ln(1-A)))
        print(cost)
        if(self.act == "sigmoid"):
            dw=(1/np.size(X,0)*np.dot((A-Y).T, X)).T
            db = 1/np.size(X,0)*np.sum(A-Y)
        else:
            dw=(1/np.size(X,0)*np.dot(np.multiply((A-np.multiply(Y,A)+np.multiply(A,A)-Y),1/(A+.001)).T, X)).T
            db = 1/np.size(X,0)*np.sum(np.multiply((A-np.multiply(Y,A)+np.multiply(A,A)-Y),1/(A+.001)))

        
        grads = [dw,db]
        return grads, cost
    
    #math function
    def sigmoid(self, z):
        s = 1/(1+np.exp(-z))
        cache = z
        return s
    
    def tanh(self, z):
        s = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        cache = z
        return s
   
    def safe_ln(self, x):
        warnings.simplefilter("error", RuntimeWarning)
        try:
           ln = np.log(x)
        except:
           ln = 0
        return ln
 
    
    #logistic regression
    def log_reg(self, w,b,X,Y,num_it,lr):
        costs = []
        for i in range(num_it):
            temp_X = random.sample(range(X.shape[0]), X.shape[0]-50)
            temp_Y = Y[temp_X[:],:]
            temp_X = X[temp_X[:],:]
            grads, cost = self.grad_cost_log_reg(w,b,temp_X,temp_Y,0)
            w = w - (1+cost)*grads[0]
            b = b - (1+cost)*grads[1]
            costs.append(cost)
        pram = [w,b]
        print(np.shape(costs))
        return pram, grads, costs
    
    
    #predict
    def pred(self, w,b,X):
        m = X.shape[0]
        Y_pred = np.zeros((1,m))
        w = w.reshape(X.shape[1],1)
        A =self.sigmoid(np.dot(X,w)+b)
        Y_pred =np.round(A)
        return Y_pred
    
    #model
    def model_log_reg(self):
        pram, grad, costs = self.log_reg(self.W, self.b, self.X.astype(float), self.Y.astype(float), self.num_it, self.lr)
        w = pram[0]
        b = pram[1]
        Y_pred_dev = self.pred(w, b, self.XC.astype(float))
        Y_pred_train = self.pred(w, b, self.X.astype(float))
        
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - self.Y)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_dev - self.YC)) * 100))
        d = {"costs": costs,
             "Y_prediction_test": Y_pred_dev, 
             "Y_prediction_train" : Y_pred_train,
             "prec_train" : 100 - np.mean(np.abs(Y_pred_train - self.Y)) * 100,
             "prec_dev": 100 - np.mean(np.abs(Y_pred_dev - self.YC)) * 100, 
             "w" : w, 
             "b" : b,
             "learning_rate" : self.lr,
             "num_iterations": self.num_it}
         
        return d
