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


num_rows = np.size(train_data, 0)
num_col = np.size(train_data,1)
Y = np.zeros((num_rows,1))
Y[:,0] = train_data[:,1]
#answeres
X = np.delete(train_data, 1, 1)
#embarked
X = np.delete(X,10,1)
#cabin
#X = np.delete(X,9,1)
#ticket
#X = np.delete(X,7,1)
#parents
#parents
#X = np.delete(X,6,1)
#names
#X = np.delete(X,2,1)
#id
X = np.delete(X,0,1)

#ENCODING_________________________________________________________________
print(np.shape(X))
enc = OneHotEncoder(handle_unknown = 'ignore')
X = enc.fit_transform(X)
X = X.toarray()
print(np.shape(X))


#LISTHUANIA___________________________________________
path = '/Users/morganharper/titanic/Listhunia/Sheet1-Table 1.csv'
LX = pd.read_csv(path)
LX = np.array(LX)
LX[:,2] = LX[:,2] + LX[:,3] + LX[:,4]
print(np.shape(LX))
LX = np.delete(LX, 4, 1)
LX = np.delete(LX, 3, 1)
LY = np.expand_dims(LX[:,0],1)
print(np.shape(LY))
LX = np.delete(LX,0,1)
np.random.shuffle(LX)
print(np.shape(LX))

#ENCODING_________________________________________________________________
LX = enc.transform(LX)
LX = LX.toarray()
print(np.shape(LX))

#PREP_______________________________________________________________________
#norm = np.linalg.norm(X)
#X = X/norm
for n in range(np.size(LX,1)):
    if np.amax(LX[:,n]) != 0:
        LX[:,n] = LX[:,n]/np.amax(LX[:,n])

#TRAIN DEV AND TEST____________________________________
LX_TRAIN = LX[0:864,:]
LY_TRAIN = LY[0:864,:]
LX_DEV = LX[864:1064,:]
LY_DEV = LY[864:1064,:]
LX_TEST = LX[1064:,:]
LY_TEST = LY[1064:,:]

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
def grad_cost_log_reg(w,b,X,Y,lambd):
    A=sigmoid(np.dot(X,w)+b)
    cases = int(np.size(X, 0))
    
    cost = -1/cases * np.sum(np.multiply(Y,np.log(A))+ np.multiply((1-Y),np.log(1-A))) + 1/cases * lambd/2 * np.sum(np.square(w))

    dw=(1/np.size(X,0)*np.dot((A-Y).T, X)).T + w*lambd/cases
    db = 1/np.size(X,0)*np.sum(A-Y)

    grads = [dw,db]
    return grads, cost

#math function
def sigmoid(z):
    #z = z.astype(float)
    s = 1/(1+np.exp(-z))
    cache = z
    return s


#logistic regression
def log_reg(w,b,X,Y,num_it,lr):

    costs = []
    for i in range(num_it):
        temp_X = random.sample(range(X.shape[0]), X.shape[0]-50)
        temp_Y = Y[temp_X[:],:]
        temp_X = X[temp_X[:],:]
        grads, cost = grad_cost_log_reg(w,b,temp_X,temp_Y,0)
        w = w - lr*grads[0]
        b = b - lr*grads[1]
        costs.append(cost)
        if(i%1000 ==0):
            print(i/1000)
    pram = [w,b]
    print(np.shape(costs))
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
    pram, grad, costs = log_reg(w,b,X_train.astype(float),Y_train.astype(float),num_it,lr)
    w = pram[0]
    b = pram[1]
    Y_pred_dev = pred(w, b, X_dev.astype(float))
    Y_pred_train = pred(w, b, X_train.astype(float))

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_dev - Y_dev)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_dev, 
         "Y_prediction_train" : Y_pred_train,
         "prec_train" : 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100,
         "prec_dev": 100 - np.mean(np.abs(Y_pred_dev - Y_dev)) * 100, 
         "w" : w, 
         "b" : b,
         "learning_rate" : lr,
         "num_iterations": num_it}

    return d

pred_train_train = []
pred_train_dev = []
pred_dev_train = []
pred_dev_dev = []
for i in range(100):
    print(i)
    w,b = inti(LX_TRAIN.shape[1])
    print("LIST TRAIN")
    l = model_log_reg(LX_TRAIN, LY_TRAIN, LX_DEV, LY_DEV, 10000*(i+1), 1.36, w, b)
    w = l["w"]
    b = l["b"]
    print("LIST DEV")
    pred_train_train.append(l["prec_train"])
    pred_train_dev.append(l["prec_dev"])
    m = model_log_reg(LX_DEV, LY_DEV, LX_TEST, LY_TEST, 10000*(i+1), 1.36, w, b)
    w = m["w"]
    b = m["b"]
    pred_dev_train.append(m["prec_train"])
    pred_dev_dev.append(m["prec_dev"])
    print("T TRAIN")
    d = model_log_reg(X_train, Y_train, X_dev, Y_dev, 10000, .5, w, b)
    w = d["w"]
    b = d["b"]
    #pred_train_train.append(d["prec_train"])
    #pred_train_dev.append(d["prec_dev"])
    print("T DEV")
    e = model_log_reg(X_dev, Y_dev, X_train, Y_train, 10000, .5, w, b)
    #pred_dev_train.append(e["prec_train"])
    #pred_dev_dev.append(e["prec_dev"])

print(max(pred_train_train))
print(pred_train_train.index(max(pred_train_train)))
print(max(pred_dev_dev))
print(pred_dev_dev.index(max(pred_dev_dev)))
print(max(pred_dev_dev + pred_train_train))
print((pred_dev_dev+pred_train_train).index(max(pred_dev_dev + pred_train_train)))

