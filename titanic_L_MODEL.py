#IMPORTS_______________________________________________________________
import numpy as np
import pandas as pd
import string
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

#DATA___________________________________________________________________
path = r'C:\Users\mharp\python\titanic\train.csv'
train = pd.read_csv(path)
train_data = np.array(train)

#MATRIX__________________________________________________________________
num_rows = np.size(train_data, 0)
num_col = np.size(train_data,1)
Y = np.zeros((num_rows,1))
Y[:,0] = train_data[:,1]
X = np.delete(train_data, 1, 1)

#ENCODING_________________________________________________________________
enc = OneHotEncoder(handle_unknown = 'ignore')
X = enc.fit_transform(X)
X = X.toarray()

#TEST AND TRAIN DATA______________________________________________________
X_train = X[0:800,:]
Y_train = Y[0:800]

X_test = X[801:-1,:]
Y_test = Y[801:-1]

#SETUP___________________________________________________________________


#set up multi-layer
def initialize_parameters_deep(layer_dims):

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

  
    return parameters


#FORWARD________________________________________________________________

#forward
def linear_forward(A,W,b):
    Z=np.dot(A,W.T) + b.T
    cache = (A,W,b)
    return Z, cache

#activation
def activation(A_prev, W, b, activation):
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b);
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b);
        A, activation_cache = relu(Z)

    elif activation == "leaky_relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b);
        A, activation_cache = leaky_relu(Z)
        
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = int(len(parameters)/2)
    check_even = 1;
    if( L%2 == 0):
        check_even = 0
    for l in range(1, L):
        A_prev = A
        A, cache = activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "leaky_relu")   
        caches.append(cache)
    AL, cache = activation(A, parameters['W'+ str(L)], parameters['b'+ str(L)], activation = "sigmoid")
    caches.append(cache)
    return AL, caches

#ACTIVATIONS______________________________________________________________

#math function
def relu(z):
    s = np.maximum(0,z)
    cache = z
    return s, cache

def leaky_relu(z):
    s = z
    s[z<=0] = s[z<=0]*.02
    cache = z
    return s, cache

#math function
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    cache = z
    return s, cache


#DIRVATIVE ACTIVATIONS______________________________________________________

#sigmoid derivative
def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.multiply(np.multiply(dA, Z),(1-Z))
    return dZ

#Relu derviative
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = dZ[Z<=0] * 0
    
    return dZ

#Leaky Relu derviative
def leaky_relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    indices = np.argwhere(Z <= 0)
    dZ[indices] = dZ[indices] * .2
    
    return dZ



#BACKWARDS__________________________________________________________________

#backward
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m * np.dot(A_prev.T, dZ)
    db = 1/m*np.matrix(np.sum(dZ, axis = 0)).T
    dA_prev = np.dot(dZ, W)
    return dA_prev, dW, db

#activation backwards
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "leaky_relu":
        dZ = leaky_relu_backwards(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

#backwards modle
def L_model_backward(AL, Y, caches, parameters):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = np.abs(AL-Y)
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#updates
def update_parameters(parameters, grads, lr):
    L = int(len(parameters)/2)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - lr*grads["dW" + str(l+1)].T
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - lr*grads["db" + str(l+1)]
    return parameters

#COSTS____________________________________________________________________________
#cost
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = .5*np.sum(np.square(AL-Y))
    cost = np.squeeze(cost)
    return cost

#MODEL____________________________________________________________________
#model
def model_L(X_train,Y_train,X_test,Y_test,layer_dims, num_it,lr):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_it):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X_train, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y_train)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y_train, caches, parameters)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, lr)
                
        # Print the cost every 100 training example
        print ("Cost after iteration %i: %f" %(i, cost))
        if i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    Y_pred_train, c = L_model_forward(X_train, parameters)
    Y_pred_test, c = L_model_forward(X_test, parameters)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_pred_test, 
         "Y_prediction_train" : Y_pred_train, 
         "pram": parameters,
         "learning_rate" : lr,

         "num_iterations": num_it}

    return d


#PREDICTION__________________________________________________________


#RUNNING_____________________________________________________________
layers_dims = [np.size(X_train,1), 500, 1000, 400, 800,1]
