
import scipy.io as io

from pylab import *
import numpy as np
import pyNN.neuron as p

import pickle

p.setup(timestep=0.1)
p.reset()
import NeuroTools.stgen as nts


# initialise DATA
data1 = io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe093-3-C3-MO.mat')['GDFcell'][0]
data2 = io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe108-7-C3-MO.mat')['GDFcell'][0]
data3 = io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe112-5-C3-MO.mat')['GDFcell'][0]
data4 = io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe112-6-C3-MO.mat')['GDFcell'][0]
data5 = io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe145-4-C3-MO.mat')['GDFcell'][0]
data = [data1, data2, data3, data4, data5]

ndirections = 6
nneurons = 5
ntrials = 30
nindependents = 5
startstimulus = 750 # TODO:
endstimulus = 1000  # TODO:
outspikes = []
inspikes = []

DATA = [[[array([]) for i in range(ntrials+nindependents)] for i in range(nneurons)] for i in range(ndirections)]
for i in range(ndirections):
    for j in range(nneurons):
        for k in range(ntrials+nindependents):
            DATA[i][j][k]=data[j][i][data[j][i][:,0]==k+1,1]
            DATA[i][j][k].sort()


# initialise neurons
dirs = range(ndirections)
npop=1


# initialise input neurons
neurons=[]
for nneuron in range(nneurons):
    neurons.append(p.Population(npop, cellclass=p.SpikeSourceArray))


# initialise output neurons
nout=2
out=[]
for i in range(nout):
    out.append(p.Population(npop, cellclass=p.IF_cond_exp))
    

# initialise connections
initweight=0.08
prj = []
for i in range(nneurons):
    for j in range(nout):
	prj.append(p.Projection(neurons[i], out[j], target="excitatory",method=p.AllToAllConnector()))
	prj[-1].setWeights(initweight)
	    
	    
def simulate():
    global direction
    global inspikes
    global outspikes
    global outspikes2
    global outvolts
    global prj
    print "Simulating..."
    # set inputs
    for ntrial in range(ntrials):
	for direction in dirs:
	    print "Training direction:", direction, ", trial:", ntrial+1
	    print "Reading data:"
	    trainingset = [DATA[direction][nneuron][ntrial] for nneuron in range(nneurons)]
	    trainingset = [trainingset[i][startstimulus < trainingset[i]] for i in range(nneurons)]
	    trainingset = [trainingset[i][trainingset[i] < endstimulus] for i in range(nneurons)]

	    # run simulation
	    p.reset()
	    for o in out:
		o.record()
		o.record_v()
	    for i, neuron in enumerate(neurons):
		neuron.set('spike_times', trainingset[i])
		#neuron[nneuron].set('spike_times',arange(1,1901,100))
		neuron.record()

	    p.run(2000)

	    outSpikeTimes = [[] for i in range(nout)]
	    outvolts = [[] for i in range(nout)]
	    ## plot spike trains
	    #fig = figure()
	    #hold(True)
	    #ax = fig.add_subplot(1,1,1)
	    #title("Direction "+str(direction)+", Trial "+str(ntrial+1))
	    for j, o in enumerate(out):
		spikes = list(o.getSpikes()[:,1])
		#print j, spikes, len(spikes), type(spikes)
		outSpikeTimes[j] = spikes
		outvolts[j] = o.get_v()
		print "--------------------------------"
		#print j, outSpikeTimes[j], len(outSpikeTimes[j])
		
		#ax.plot(outSpikeTimes[j], [j]*len(outSpikeTimes[j]), 'b|', markersize = 20.)

	    inspikes=[0 for i in range(nneurons)]
	    outspikes=[0 for i in range(nout)]
	    outspikes2=[]
	    outvolts=[]
	    for i,o in enumerate(out):
		outspikes[i] = o.get_spike_counts().values()[0]
		#outspikes2.append(o.getSpikes())
		outvolts.append(o.get_v())

	    for i,neuron in enumerate(neurons):
		inspikes[i] = neurons[i].get_spike_counts().values()[0]
		#print inspikes[i]

	    def learn():
		global direction
		global inspikes
		global outspikes
		global outspikes2
		global outvolts
		print "Learning..."
		#-----------------------------------------------------
		def updateWeights():
		    global direction
		    global inspikes
		    global outspikes
		    global outspikes2
		    global outvolts
		    global prj
		    print "outspikes:", outspikes
		    print "Updating..."
		    adjust=0.02
		    negadjust=0.02
		    nmax=inspikes.index(max(inspikes))
		    if (outspikes[0]<outspikes[1]) and (direction==dirs[1]):     # FIXME:
			prj[2*nmax+1].setWeights(prj[2*nmax+1].getWeights()[0]+adjust)
			print 'correct 0'
			print "Updated to:", prj[2*nmax+1].getWeights()
		    elif (outspikes[0]>outspikes[1]) and (direction==dirs[0]):   # FIXME:
			prj[2*nmax+0].setWeights(prj[2*nmax+0].getWeights()[0]+adjust)
			print 'correct 3'                                            # FIXME:
			print "Updated to:", prj[2*nmax+0].getWeights()
		    elif (outspikes[0]>=outspikes[1]) and (direction==dirs[1]):  # FIXME:
			print 'wrong 0'
			prj[2*nmax+0].setWeights(max(0,prj[2*nmax+0].getWeights()[0]-negadjust))
			print "Updated to:", prj[2*nmax+0].getWeights()
		    elif (outspikes[0]<=outspikes[1]) and (direction==dirs[0]):  # FIXME:
			prj[2*nmax+1].setWeights(max(0,prj[2*nmax+1].getWeights()[0]-negadjust))
			print 'wrong 3' 
			print "Updated to:", prj[2*nmax+1].getWeights()
		    else:
			print 'no 5'
		    print
		updateWeights()
		#-----------------------------------------------------
		for p in prj:
		    currentWeight = p.getWeights()[0]
		    print direction, ntrial, p, currentWeight

	    learn()
    goodWeights = [pr.getWeights() for i, pr in enumerate(prj)]
    print "goodWeights:", goodWeights 
    pickle.dump(goodWeights, open("goodWeightsFile.pkl", "wb"))

def validate():
    global direction
    global inspikes
    global outspikes
    global outspikes2
    global outvolts
    global prj
    
    prj = []
    for i in range(nneurons):
	for j in range(nout):
	    prj.append(p.Projection(neurons[i], out[j], target="excitatory",method=p.AllToAllConnector()))
	    prj[-1].setWeights(initweight)
    goodWeights = pickle.load(open("goodWeightsFile.pkl"))
    for i, pr in enumerate(prj):
	print goodWeights[i], pr
	pr.setWeights(goodWeights[i])
    print "Validating..."
    # set inputs
    for ntrial in range(ntrials, ntrials+nindependents):
	for direction in dirs:
	    print "Checking direction:", direction, ", trial:", ntrial+1
	    print "Reading data:"
	    trainingset = [DATA[direction][nneuron][ntrial] for nneuron in range(nneurons)]
	    trainingset = [trainingset[i][startstimulus < trainingset[i]] for i in range(nneurons)]
	    trainingset = [trainingset[i][trainingset[i] < endstimulus] for i in range(nneurons)]
	    import pdb; pdb.set_trace()

	    # run simulation
	    p.reset()
	    for o in out:
		o.record()
		o.record_v()
	    for i, neuron in enumerate(neurons):
		neuron.set('spike_times', trainingset[i])
		#neuron[nneuron].set('spike_times',arange(1,1901,100))
		neuron.record()

	    p.run(2000)

	    outSpikeTimes = [[] for i in range(nout)]
	    outvolts = [[] for i in range(nout)]
	    ## plot spike trains
	    #fig = figure()
	    #hold(True)
	    #ax = fig.add_subplot(1,1,1)
	    #title("Direction "+str(direction)+", Trial "+str(ntrial+1))
	    for j, o in enumerate(out):
		spikes = list(o.getSpikes()[:,1])
		#print j, spikes, len(spikes), type(spikes)
		outSpikeTimes[j] = spikes
		outvolts[j] = o.get_v()
		#print j, outSpikeTimes[j], len(outSpikeTimes[j])
		
		#ax.plot(outSpikeTimes[j], [j]*len(outSpikeTimes[j]), 'b|', markersize = 20.)

	    inspikes=[0 for i in range(nneurons)]
	    outspikes=[0 for i in range(nout)]
	    outspikes2=[]
	    outvolts=[]
	    for i,o in enumerate(out):
		outspikes[i] = o.get_spike_counts().values()[0]
		#outspikes2.append(o.getSpikes())
		outvolts.append(o.get_v())
	    print "outspikes:", outspikes

	    for i,neuron in enumerate(neurons):
		inspikes[i] = neurons[i].get_spike_counts().values()[0]
		#print inspikes[i]

	    def check():
		global direction
		global inspikes
		global outspikes
		global outspikes2
		global outvolts
		print "Checking..."
		#-----------------------------------------------------
		def printWeights():
		    global direction
		    global inspikes
		    global outspikes
		    global outspikes2
		    global outvolts
		    global prj
		    print "outspikes:", outspikes
		    adjust=0.02
		    negadjust=0.02
		    nmax=inspikes.index(max(inspikes))
		    if (outspikes[0]<outspikes[1]) and (direction==dirs[1]):      # FIXME:
			print 'correct 0'
		    elif (outspikes[0]>outspikes[1]) and (direction==dirs[0]):    # FIXME:
			print 'correct 3'
		    elif (outspikes[0]>=outspikes[1]) and (direction==dirs[1]):   # FIXME:
			print 'wrong 0'
		    elif (outspikes[0]<=outspikes[1]) and (direction==dirs[0]):   # FIXME:
			print 'wrong 3' 
		    else:
			print 'no 5'
		    print
		printWeights()
		#-----------------------------------------------------
		for p in prj:
		    currentWeight = p.getWeights()[0]
		    print direction, ntrial, p, currentWeight

	    check()


print "--------------------------------"
print "Simulation..."
#simulate()
print "--------------------------------"
print "Validation..."
validate()

#show()
#import pdb; pdb.set_trace()
