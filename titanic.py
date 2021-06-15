import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from act import act
from safe import safe
from costs import costs
import warnings

class InvalidActivation(Exception):
    """Invalid Activation"""

class InvalidCost(Exception):
    """Invalid Cost"""

class titanic:
    def __init__(self, X, Y,CompX, CompY, num_it, lr, W, b, activation, c):
        self.X = X
        self.Y = Y
        self.XC = CompX
        self.YC = CompY
        self.num_it = num_it
        self.lr = lr
        self.W = W
        print(W)
        self.b = b
        self.actvation = activation
        self.c = c
        self.act = act()
        self.costs = costs()
        self.safe = safe()
    
    #cost function
    def grad_cost_log_reg(self, X,Y,lambd):
        if(self.actvation == "sigmoid"):
            A=self.act.sigmoid(np.dot(X,self.W)+self.b)
            print(A)
        elif(self.actvation == "tanh"):
            A = self.act.tanh(np.dot(X,self.W) + self.b)
        elif(self.actvation == "relu"):
            A = self.act.relu(np.dot(X,self.W) + self.b)
        elif(self.actvation == "leaky_relu"):
            A = self.act.leaky_relu(np.dot(X,self.W) + self.b)
        elif(self.actvation == "exp_relu"):
            A = self.act.exp_relu(np.dot(X,self.W) + self.b)
        elif(self.actvation == "swish"):
            A = self.act.swish(np.dot(X,self.W) + self.b)
        elif(self.actvation == "softplus"):
            A = self.act.softplus(np.dot(X,self.W) + self.b)
        elif(self.actvation == "gaussian"):
            A = self.act.gaussian(np.dot(X,self.W) + self.b)
        else:
            raise InvalidActivation
        cases = int(np.size(X, 0))
        if(self.c == "quad"):
            cost = self.costs.quad(A, Y, cases)
            prime  = self.costs.quad_prime(A,Y)
        elif(self.c == "cross_entropy"):
            cost = self.costs.cross_entropy(A, Y, cases)
            prime = self.costs.cross_entropy_prime(A, Y)
        elif(self.c == "expc"):
            cost = self.costs.expc(A, Y, 1, cases)
            prime = self.costs.expc_prime(A, Y, 1, cases)
        elif(self.c == "hell"):
            cost = self.costs.hell(A, Y, cases)
            prime = self.costs.hell_prime(A, Y)
        elif(self.c == "KLD"):
            cost = self.costs.KLD(A, Y, cases)
            prime = self.costs.KLD_prime(A, Y)
        elif(self.c == "ISD"):
            cost = self.costs.ISD(A, Y, cases)
            prime = self.costs.ISD_prime(A, Y)
        else:
            raise InvalidCost
        if(self.actvation == "sigmoid"):
            dw=1/cases*np.dot(np.multiply(self.act.sigmoid_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.sigmoid_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "tanh"):
            dw=np.dot(np.multiply(self.act.tanh_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.tanh_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "relu"):
            dw=np.dot(np.multiply(self.act.relu_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.relu_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "leaky_relu"):
            dw=np.dot(np.multiply(self.act.leaky_relu_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.leaky_relu_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "exp_relu"):
            dw=np.dot(np.multiply(self.act.exp_relu_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.exp_relu_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "swish"):
            dw=np.dot(np.multiply(self.act.swish(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.swish(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "softplus"):
            dw=np.dot(np.multiply(self.act.softplus_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.softplus_prime(np.dot(X,self.W)+self.b), prime))
        elif(self.actvation == "gaussian"):
            dw=np.dot(np.multiply(self.act.gaussian_prime(np.dot(X,self.W)+self.b), prime).T, X).T
            db = 1/cases*np.sum(np.multiply(self.act.gaussian_prime(np.dot(X,self.W)+self.b), prime))
        grads = [dw,db]
        return grads, cost
 
    
    #logistic regression
    def log_reg(self, X, Y, num_it,lr):
        costs = []
        for i in range(num_it):
            temp_X = random.sample(range(X.shape[0]), X.shape[0]-50)
            temp_Y = Y[temp_X[:],:]
            temp_X = X[temp_X[:],:]
            grads, cost = self.grad_cost_log_reg(temp_X,temp_Y,0)
            print(cost)
            self.W = self.W - grads[0]*lr
            self.b = self.b - grads[1]*lr
            costs.append(cost)
        return grads, costs
    
    
    #predict
    def pred(self, X):
        m = X.shape[0]
        Y_pred = np.zeros((1,m))
        w = self.W.reshape(X.shape[1],1)
        A =self.act.sigmoid(np.dot(X,w)+self.b)
        Y_pred =np.round(A)
        return Y_pred
    
    #model
    def model_log_reg(self):
        grad, costs = self.log_reg(self.X.astype(float), self.Y.astype(float), self.num_it, self.lr)
        Y_pred_dev = self.pred(self.XC.astype(float))
        Y_pred_train = self.pred(self.X.astype(float))
        
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - self.Y)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_dev - self.YC)) * 100))
        d = {"costs": costs,
             "Y_prediction_test": Y_pred_dev, 
             "Y_prediction_train" : Y_pred_train,
             "prec_train" : 100 - np.mean(np.abs(Y_pred_train - self.Y)) * 100,
             "prec_dev": 100 - np.mean(np.abs(Y_pred_dev - self.YC)) * 100, 
             "w" : self.W, 
             "b" : self.b,
             "learning_rate" : self.lr,
             "num_iterations": self.num_it}
         
        return d
