
import random
from math import exp

class LogSigmoid(object):
	'''Proof of concept'''

	def __init__(self, center, scale):
		self.center = center
		self.scale = scale

	def logSigm(self, x):
		return 1/(1+exp(-self.scale*x-self.center))
  
if __name__ == '__main__':
	myinstance = LogSigmoid(7, 8)
	print myinstance.logSigm(1)

	otherinstance = LogSigmoid(0, 1)
	print otherinstance.logSigm(1)
