from safe import safe
from act import act
from costs import costs
import numpy as np

a = act()
c = costs()
z = safe()

def sigmoid(z):
        s = 1/(1+np.exp(-z))
        cache = z
        return s

X = np.random.rand(5, 5)
W = np.random.rand(5,1)
Y = np.random.rand(5,1)

A1 = sigmoid(np.dot(X,W))
A2 = a.sigmoid(np.dot(X,W))

print("A diff")
print(A1-A2)

print("cost diff")
print(-1/np.size(X,0) * np.sum(np.multiply(Y,z.ln(A1))+ np.multiply((1-Y),z.ln(1-A1)))- c.cross_entropy(A2, Y, np.size(X,0)))

prime = c.cross_entropy_prime(A2, Y)
print((1/np.size(X,0)*np.dot((A1-Y).T, X)).T-1/np.size(X,0)*np.dot(np.multiply(a.sigmoid_prime(np.dot(X,W)), prime).T, X).T)
