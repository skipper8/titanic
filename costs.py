#Costs
#has cost function and their dirvatives

#IMPORTS
import numpy as np
from safe import safe
import math

class costs():
    def __init__(self):
        self.safe = safe()

    def quad(self, x, y, cases):
        return np.sum(np.square(x-y))/2/cases

    def quad_prime(self, x, y):
        return abs(x-y)

    def cross_entropy(self, x, y, cases):
        return np.sum(np.multiply(y, self.safe.ln(x)) + np.multiply(1-y, self.safe.ln(1-x)))/cases

    def cross_entropy_prime(self, x, y):
        return np.divide(x-y, np.multiply(1-x+.00000001, x+.000000001))

    def expc(self, x, y, t, cases):
        return t*self.safe.exp(2*self.qaud(x,y, cases)/t)

    def expc_prime(self, x, y, t, cases):
        return 2/t*self.quad_prime(x,y)*expc(x, y, t, cases)

    def hell(self, x, y, cases):
        return 1/math.sqrt(2)*np.sum(np.square(np.sqrt(x)-np.sqrt(y)))

    def hell_prime(self, x, y):
        return np.divide(np.sqrt(x)-np.sqrt(y), math.sqrt(2)*np.sqrt(x))
    
    def KLD(self, x, y, cases):
        return (np.sum(np.multiply(y, self.safe.ln(np.divide(y,x+.0000001)))-np.sum(y) + np.sum(x)))/cases
    def KLD_prime(self, x, y):
        return np.divide(x-y, x+.00000001)
    
    def ISD(self, x, y, cases):
        return (np.sum(cp.divide(y, x+.00000001)-self.safe.ln(np.divide(y, x+.000000001))-1))/cases
    
    def ISD_prime(self, x, y):
        return np.divide(x-y, np.square(x)+.00000001)

