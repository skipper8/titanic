#test possible models

#Libraries________________________________________________________________________
import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import random
from titanic import titanic

#DATA_TITANIC________________________________________
#where file is stored
path = '/Users/morganharper/titanic/train.csv'
#get file data
train = pd.read_csv(path)
#make data array
train_data = np.array(train)

#get array dimensions
num_rows = np.size(train_data, 0)
num_col = np.size(train_data,1)

#labeling
Y = np.zeros((num_rows,1))
Y[:,0] = train_data[:,1]
X = np.delete(train_data, 1, 1)

#columns considered for removal
#embarked
X = np.delete(X,10,1)
#cabin
X = np.delete(X,9,1)
#ticket
#X = np.delete(X,7,1)
#parents
#parents
#X = np.delete(X,6,1)
#names
X = np.delete(X,2,1)
#id
X = np.delete(X,0,1)

X[:,0] = np.square(X[:,0])

#ENCODING_________________________________________________________________
enc = OneHotEncoder(handle_unknown = 'ignore')
X = enc.fit_transform(X)
X = X.toarray()

#get array dimensions
num_rows = np.size(train_data, 0)
num_col = np.size(train_data,1)

#PREP_______________________________________________________________________
#norm = np.linalg.norm(X)
#X = X/norm
for n in range(num_col):
    if np.amax(X[:,n]) != 0:
        X[:,n] = X[:,n]/np.amax(X[:,n])

#DEV AND TRAIN DATA______________________________________________________
X_train = X[0:600,:]
X_train = X_train #/np.max(X_train, axis = 0)
Y_train = Y[0:600]

X_dev = X[601:800,:]
X_dev = X_dev #/np.max(X_train, axis = 0)
Y_dev = Y[601:800]

X_test = X[801:-1,:]
X_test = X_test#/np.max(X_dev, axis = 0)
Y_test = Y[801:-1]

#TEST_FINAL______________________________________________________
path = '/Users/morganharper/titanic/test.csv'
tf_data = pd.read_csv(path)
tf = np.array(tf_data)

#columns considered for removal
#embarked
tf = np.delete(tf,10,1)
#cabin
tf = np.delete(tf,9,1)
#ticket
#tf = np.delete(tf,7,1)
#parents
#parents
#tf = np.delete(tf,6,1)
#names
tf = np.delete(tf,2,1)
#id
tf = np.delete(tf,0,1)

tf[:,0] = np.square(tf[:,0])

#ENCODING_________________________________________________________________
tf = enc.transform(tf)
X_tf = tf.toarray()

#Prep_______________________________________________________________________
#norm = np.linalg.norm(X_tf)
#X_tf = X_tf/norm
for n in range(num_col-1):
    if np.amax(X_tf[:,n]) != 0:
        X_tf[:,n] = X_tf[:,n]/np.amax(X_tf[:,n])

#DATA_LISTHUANIA___________________________________________
#find file
path = '/Users/morganharper/titanic/Listhunia/Sheet1-Table 1.csv'
#get file data
LX = pd.read_csv(path)
#make data array
LX = np.array(LX)

#name fix
LX[:,2] = LX[:,2] + LX[:,3] + LX[:,4]
LX = np.delete(LX, 4, 1)
LX = np.delete(LX, 3, 1)

LX = np.delete(LX, 9, 1)
#LX = np.delete(LX, 7, 1)
LX = np.delete(LX, 2, 1)

#labeling
LY = np.expand_dims(LX[:,0],1)
LX = np.delete(LX,0,1)

LX[:,0] = np.square(LX[:,0])


#shuffel data
np.random.shuffle(LX)
print(np.shape(LX))

#ENCODING_________________________________________________________________
LX = enc.transform(LX)
LX = LX.toarray()

#PREP_______________________________________________________________________
#norm = np.linalg.norm(X)
#LX = LX/norm
for n in range(num_col):
    if np.amax(LX[:,n]) != 0:
        LX[:,n] = LX[:,n]/np.amax(LX[:,n])

#TRAIN DEV AND TEST____________________________________
LX_TRAIN = LX[0:864,:]
LY_TRAIN = LY[0:864,:]
LX_DEV = LX[864:1064,:]
LY_DEV = LY[864:1064,:]
LX_TEST = LX[1064:,:]
LY_TEST = LY[1064:,:]


#setup
def inti(a):
    w = np.zeros((a,1))
    b = 0
    return w,b

pred_train_train = []
pred_train_dev = []
pred_dev_train = []
pred_dev_dev = []
fig, axs = plt.subplots(8)
w,b = inti(LX_TRAIN.shape[1])
print("LIST TRAIN")
list_train = titanic(LX, LY, X_train, Y_train, 50, 5, w, b, "sigmoid")
l = list_train.model_log_reg()
w = l["w"]
b = l["b"]
axs[0].plot(range(len(l["costs"])), l["costs"])
print("T TRAIN")
T_TRAIN = titanic(X_train, Y_train, X_dev, Y_dev, 300, .5, w, b, "sigmoid")
d = T_TRAIN.model_log_reg()
w = d["w"]
b = d["b"]
axs[1].plot(range(len(d["costs"])), d["costs"])
#pred_train_train.append(d["prec_train"])
#pred_train_dev.append(d["prec_dev"])
print("T DEV")
T_DEV = titanic(X_dev, Y_dev, X_test, Y_test, 250, 5, w, b, "sigmoid")
e = T_DEV.model_log_reg()
w = e["w"]
b = e["b"]
axs[2].plot(range(len(e["costs"])), e["costs"])
print("T test")
T_TEST = titanic(X_test, Y_test, X_train, Y_train, 250, 5, w, b, "sigmoid")
f = T_TEST.model_log_reg()
axs[3].plot(range(len(f["costs"])), f["costs"])
print("tanh")
w,b = inti(LX_TRAIN.shape[1])
print("LIST TRAIN")
list_train = titanic(LX, LY, X_train, Y_train, 50, 5, w, b, "tanh")
r = list_train.model_log_reg()
w = l["w"]
b = l["b"]
axs[4].plot(range(len(l["costs"])), l["costs"])
print("T TRAIN")
T_TRAIN = titanic(X_train, Y_train, X_dev, Y_dev, 300, .5, w, b, "tanh")
s = T_TRAIN.model_log_reg()
w = d["w"]
b = d["b"]
axs[5].plot(range(len(d["costs"])), d["costs"])
#pred_train_train.append(d["prec_train"])
#pred_train_dev.append(d["prec_dev"])
print("T DEV")
T_DEV = titanic(X_dev, Y_dev, X_test, Y_test, 250, 5, w, b, "tanh")
t = T_DEV.model_log_reg()
w = e["w"]
b = e["b"]
axs[6].plot(range(len(e["costs"])), e["costs"])
print("T test")
T_TEST = titanic(X_test, Y_test, X_train, Y_train, 250, 5, w, b, "tanh")
u = T_TEST.model_log_reg()
axs[7].plot(range(len(f["costs"])), f["costs"])

plt.show()
