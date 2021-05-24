#SAFE LN
#safe version of np log and exp

#IMPORTS
import numpy as np
import warnings

class safe():
    def __init__(self):
        warnings.simplefilter("error", RuntimeWarning)

    def ln(self, x):
        
        for i in x:
            for j in i:
                try:
                    j = np.log(j)
                except:
                    if( j>0):
                         j = 10,000
                    else:
                         j = -10,000
        return x

    def exp(self, x):
        for i in x:
            for j in i:
                try:
                    j = np.exp(j)
                except:
                    if( j>0):
                         j = 10,000
                    else:
                         j = .01
        return x
