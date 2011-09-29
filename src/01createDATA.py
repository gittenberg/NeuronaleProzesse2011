
import scipy.io as io

from pylab import *
import numpy
import pyNN.neuron as p

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
ntrials = 2 #TODO: 35

DATA = [[[array([]) for i in range(ntrials)] for i in range(nneurons)] for i in range(ndirections)]
for i in range(ndirections):
    for j in range(nneurons):
        for k in range(ntrials):
            DATA[i][j][k]=data[j][i][data[j][i][:,0]==k+1,1]
            DATA[i][j][k].sort()


# initialise neurons
dir_1 = 0
dir_2 = 3
dirs = [dir_1, dir_2]
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
initweight=0.05
prj=[]
for i in range(nneurons):
    for j in range(nout):
        prj.append(p.Projection(neurons[i], out[j], target="excitatory",method=p.AllToAllConnector()))
        prj[-1].setWeights(initweight)
        
        
# set inputs
for direction in dirs:
    for ntrial in range(ntrials):
	print "Training direction:", direction, ", trial:", ntrial+1
	print "Reading data:"
	trainingset = [DATA[direction][nneuron][ntrial] for nneuron in range(nneurons)] #TODO: select time window
	#print trainingset

	# run simulation
	p.reset()
	for o in out:
	    o.record()
	    o.record_v()
	for i, neuron in enumerate(neurons):
	    neuron.set('spike_times', trainingset[i])
	    #neuron[nneuron].set('spike_times',arange(1,1901,100))

	p.run(2000)

	outspikes = [[] for i in range(nout)]
	outvolts = [[] for i in range(nout)]
	print "--------------------------------"
	# plot spike trains
	fig = figure()
	hold(True)
	ax = fig.add_subplot(1,1,1)
	title("Direction "+str(direction)+", Trial "+str(ntrial+1))
	for j, o in enumerate(out):
	    spikes = list(o.getSpikes()[:,1])
	    #print j, spikes, len(spikes), type(spikes)
	    outspikes[j] = spikes
	    outvolts[j] = o.get_v()
	    print "--------------------------------"
	    print j, outspikes[j], len(outspikes[j])
	    
	    ax.plot(outspikes[j], [j]*len(outspikes[j]), 'b|', markersize = 20.)

	    def learn():
		print "Learning..."

	    learn()



show()
#import pdb; pdb.set_trace()


    
