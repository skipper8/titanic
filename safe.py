#SAFE LN
#safe version of np log and exp

#IMPORTS
import numpy as np
import warnings

class safe():
    def __init__(self):
        warnings.simplefilter("error", RuntimeWarning)

    def ln(self, x):
        try:
            x = np.log(x)
        except:
            x = 0
        return x

    def exp(self, x):
        try:
            x = np.exp(x)
        except:
            x = 0
        return x
