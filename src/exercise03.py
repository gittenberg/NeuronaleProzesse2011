
import random
from math import exp

if __name__ == '__main__':
    s = '432101'
    l = len(s)
    base = 8
    
    total = 0
    for i, j in enumerate(s):
      total += int(j)*base**(l-i-1)
    print total
    
    tmp = [base**(l-i-1)*int(j) for (i, j) in enumerate(s)]
    print sum(tmp)
    print reduce(lambda x, y: x+y, tmp)