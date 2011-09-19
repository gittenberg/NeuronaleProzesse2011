
import random
from math import exp

def logSigm(x):
    return 1/(1+exp(x))
  
if __name__ == '__main__':
    x = 7
    print x, '\t', logSigm(x)
    
    y = random.random()
    print y, '\t', logSigm(y)
