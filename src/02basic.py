
import scipy.io as io

from pylab import *
import numpy
import pyNN.neuron as p

p.setup(timestep=0.1)
p.reset()
import NeuroTools.stgen as nts


data1=io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe093-3-C3-MO.mat')['GDFcell'][0]
data2=io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe108-7-C3-MO.mat')['GDFcell'][0]
data3=io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe112-5-C3-MO.mat')['GDFcell'][0]
data4=io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe112-6-C3-MO.mat')['GDFcell'][0]
data5=io.loadmat('/home/bude/k0018000/NEUROprojekt/bootstrap_joe145-4-C3-MO.mat')['GDFcell'][0]
data=[data1,data2,data3,data4,data5]

ndirections = 6
nneurons = 5
ntrials = 35
DATA = [[[array([]) for i in range(ntrials)] for i in range(nneurons)] for i in range(ndirections)]
  

 
for i in range(ndirections):
    for j in range(nneurons):
        for k in range(ntrials):
            DATA[i][j][k]=data[j][i][data[j][i][:,0]==k+1,1]
            DATA[i][j][k].sort()
            

#import pdb; pdb.set_trace()


dirs = [0,3]
npop=1
neuron=[]
for nneuron in range(nneurons):
    neuron.append(p.Population(npop, cellclass=p.SpikeSourceArray))

#nneuron=3



nout=2
out=[]
for i in range(nout):
    out.append(p.Population(npop, cellclass=p.IF_cond_exp))
    

initweight=0.08
prj=[]
for i in range(nneurons):
    for j in range(nout):
        prj.append(p.Projection(neuron[i], out[j], target="excitatory",method=p.AllToAllConnector()))
        prj[-1].setWeights(initweight)


for o in out:
    o.record()
    o.record_v()


for n in neuron:
  n.record()
  
  
direction=-1  
inspikes=[0,0,0,0,0]
outspikes=[0,0]
outspikes2=[]
outvolts=[]

def sim(trial):
    global direction
    direction=0
    print direction
    p.reset()
    #for direction in dirs:
    for n in range(len(neuron)):
      neuron[n].set('spike_times',DATA[direction][n][trial])

    p.run(2000)

    outspikes=[0,0]
    outspikes2=[]
    outvolts=[]
    for i,o in enumerate(out):
        outspikes[i]=o.get_spike_counts().values()[0]
        outspikes2.append(o.getSpikes())
        outvolts.append(o.get_v())
    inspikes=[0,0,0,0,0]

    for i,n in enumerate(neuron):
        inspikes[i]=n.get_spike_counts().values()[0]
        
    """  
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    hold(True)
    for i in range(nout):
        ax.plot(outspikes2[i][:,1],i*ones_like(outspikes2[i][:,1]),'b|')
    ax.set_ylim(-6,5)
    for i in range(nneurons):
   
#    ax.plot(DATA[direction][i][trial],-1-i*ones_like(DATA[direction][i][trial]),'r|')
        inspikes2=neuron[i].getSpikes()
        ax.plot(inspikes2,-1-i*ones_like(inspikes2),'r|')
    #ax2=fig.add_subplot(2,1,2)
    #ax2.plot(outvolts[0][:,1],outvolts[0][:,2])
    """
    return inspikes,outspikes

def updateWeights():
    adjust=0.1
    negadjust=0.02
    nmax=inspikes.index(max(inspikes))
    if (outspikes[0]<outspikes[1]) and (direction==3):
        prj[2*nmax+1].setWeights(prj[2*nmax+1].getWeights()[0]+adjust)
        print 'correct'
    elif (outspikes[0]>outspikes[1]) and (direction==0): 
        prj[2*nmax+0].setWeights(prj[2*nmax+0].getWeights()[0]+adjust)
	print 'correct'
    elif (outspikes[0]>=outspikes[1]) and (direction==3):
	prj[2*nmax+0].setWeights(max(0,prj[2*nmax+0].getWeights()[0]-negadjust))
    elif (outspikes[0]<=outspikes[1]) and (direction==0):
	 print 'wrong'
	 prj[2*nmax+1].setWeights(max(0,prj[2*nmax+1].getWeights()[0]-negadjust))
	 print 'wrong' 
    else:
      
	 print 'no'
	
    
  
def train():
    for i in range(10):
        sim(i)
        updateWeights()
      
      
    
   
    
    