# test from laptop
# test from Windows


#neuron.load_mechanisms('/usr/local/lib/python2.7/dist-packages/pyNN/neuron/nmodl')

import pyNN
import inspect

print(pyNN.__file__)

import pyNN.neuron as p

#from pyNN.neuron import * #doctest: +SKIP
from pyNN.nest import *  #doctest: +SKIP
#from pyNN.pcsim import *  #doctest: +SKIP
#from pyNN.brian import *  #doctest: +SKIP

print "hallo welt"