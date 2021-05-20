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
            if( x.all()>0):
              x = 10,000
            else:
              x = -10,000
        return x

    def exp(self, x):
        try:
           x = np.exp(x)
        except:
           if(x.all() > 0):
               x = 22000
           else:
               x = 0
        return x
