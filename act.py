#activations
#
#holds sigmoid function and dirivative used often as an activations function

#IMPORTS
import numpy as np
from safe import safe

class act():
    def __init__(self):
        self.safe = safe()
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_prime(self, x):
        return np.multiply(self.sigmoid(x), 1-self.sigmoid(x))

    def safe_sigmoid(self, x):
        return 1/(1+self.safe.exp(-x))

    def safe_sigmoid_prime(self, x):
        return np.multiply(self.safe_sigmoid(x), 1-self.safe_sigmoid(x))

    def tanh(self, x):
        return np.divide(self.safe.exp(x)-self.safe.exp(-x), self.safe.exp(x) + self.safe.exp(-x)+.000001)

    def tanh_prime(self, x):
        return 1-np.square(self.tanh(x))

    def relu(self,x,m):
        return np.maximum(m, x)

    def relu_prime(self,x,m):
        temp_a = np.where(x <= m, 0, x)
        return np.where(x > m, 1, temp_a)

    def leaky_relu(self,x,m, n):
        return np.where(x <= m, x*n, x)

    def leaky_relu_prime(self,x,m,n):
        temp_a = np.where(x <= m, n, x)
        return np.where(x > m, 1, temp_a)

    def exp_relu(self,x,m, n):
        return np.where(x <= m, (self.safe.exp(x)-1)*n, x)

    def exp_relu_prime(self,x,m,n):
        temp_a = np.where(x <= m, n*self.safe.exp(x), x)
        return np.where(x > m, 1, temp_a)

    def swish(self, x):
        return np.multiply(x, self.sigmoid(x))

    def swish_prime(self, x):
        return np.multiply(x,self.sigmoid_prime(x))+self.sigmoid(x)

    def softplus(self, x):
        return self.safe.ln(1+self.safe.exp(x))
    
    def softplus_prime(self, x):
        return self.sigmoid(x)

    def gaussian(self, x):
        return self.safe.exp(-1*np.square(x))

    def gaussian_prime(self, x):
        return -2*np.multiply(x,self.gaussian(x))
