import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random

#TRAIN AND DEV_________________________________________
path = '/Users/morganharper/titanic/train.csv'
train = pd.read_csv(path)
train_data = np.array(train)
print(train_data[:,8])

num_rows = np.size(train_data, 0)
num_col = np.size(train_data,1)
Y = np.zeros((num_rows,1))
Y[:,0] = train_data[:,1]
#answeres
X = np.delete(train_data, 1, 1)
#cabins
#X = np.delete(X,7,1)
#names
X = np.delete(X,2,1)
#id
X = np.delete(X,0,1)
print(X)

#ENCODING_________________________________________________________________
enc = OneHotEncoder(handle_unknown = 'ignore')
X = enc.fit_transform(X)
X = X.toarray()

#PREP_______________________________________________________________________
#norm = np.linalg.norm(X)
#X = X/norm
for n in range(num_col):
    if np.amax(X[:,n]) != 0:
        X[:,n] = X[:,n]/np.amax(X[:,n])

#DEV AND TRAIN DATA______________________________________________________
X_train = X[0:800,:]
X_train = X_train #/np.max(X_train, axis = 0)
Y_train = Y[0:800]

X_dev = X[801:-1,:]
X_dev = X_dev#/np.max(X_dev, axis = 0)
Y_dev = Y[801:-1]

#TEST_______________________________________________________
path = '/Users/morganharper/titanic/test.csv'
test_data = pd.read_csv(path)
test = np.array(test_data)

#ENCODING_________________________________________________________________
enc = OneHotEncoder(handle_unknown = 'ignore')
X = enc.fit_transform(test)
X_test = X.toarray()

#Prep_______________________________________________________________________
#norm = np.linalg.norm(X_test)
#X_test = X_test/norm
for n in range(num_col):
    if np.amax(X_test[:,n]) != 0:
        X_test[:,n] = X_test[:,n]/np.amax(X_test[:,n])

#setup
def inti(a):
    w = np.zeros((a,1))
    b = 0
    return w,b

#cost function
def grad_cost_log_reg(w,b,X,Y):
    A=sigmoid(np.dot(X,w)+b)
    cases = int(np.size(X, 0))
    
    cost = -1/cases * np.sum(np.multiply(Y,np.log(A))+ np.multiply((1-Y),np.log(1-A)))

    dw=(1/np.size(X,0)*np.dot((A-Y).T, X)).T
    db = 1/np.size(X,0)*np.sum(A-Y)

    grads = [dw,db]
    return grads, cost

#math function
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    cache = z
    return s


#logistic regression
def log_reg(w,b,X,Y,num_it,lr):

    costs = []
    for i in range(num_it):
        temp_X = random.sample(range(X.shape[0]), X.shape[0]-5)
        temp_Y = Y[temp_X[:],:]
        temp_X = X[temp_X[:],:]
        grads, cost = grad_cost_log_reg(w,b,temp_X,temp_Y)
        w = w - lr*grads[0]
        b = b - lr*grads[1]
        costs.append(cost)
        if(i%1000 ==0):
            print(i/1000)
    pram = [w,b]
    return pram, grads, costs
    

#predict
def pred(w,b,X):
    m = X.shape[0]
    Y_pred = np.zeros((1,m))
    w = w.reshape(X.shape[1],1)
    A =sigmoid(np.dot(X,w)+b)
    Y_pred =np.round(A)
    return Y_pred

#model
def model_log_reg(X_train,Y_train,X_dev,Y_dev,num_it,lr, w, b):  
    pram, grad, costs = log_reg(w,b,X_train,Y_train,num_it,lr)
    w = pram[0]
    b = pram[1]
    Y_pred_dev = pred(w, b, X_dev)
    Y_pred_train = pred(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_dev - Y_dev)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_dev, 
         "Y_prediction_train" : Y_pred_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : lr,
         "num_iterations": num_it}

    return d
    
w,b = inti(X_train.shape[1])
d = model_log_reg(X_train, Y_train, X_dev, Y_dev, 100000, .01,w,b)
w = d["w"]
b = d["b"]
e = model_log_reg(X_dev, Y_dev, X_train, Y_train, 10000, .01, w, b)
