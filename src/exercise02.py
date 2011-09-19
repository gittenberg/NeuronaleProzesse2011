
import random
from math import exp

class LogSigmoid(object):
	'''Proof of concept'''

	def __init__(self, center, scale):
		self.center = center
		self.scale = scale

	def logSigm(self, x):
		return 1/(1+exp(-self.scale*x-self.center))
		
	def listLogSigm(self, li):
		return [self.logSigm(x) for x in li]
  
if __name__ == '__main__':
	myinstance = LogSigmoid(7, 8)
	print myinstance.logSigm(1)

	otherinstance = LogSigmoid(0, 1)
	print otherinstance.logSigm(1)
	
	examplelist = [1, 2, 3]
	print otherinstance.listLogSigm(examplelist)
