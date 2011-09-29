
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


dir_1 = 0
dir_2 = 3
dirs = [dir_1, dir_2]
npop=1
neuron=[]
for nneuron in range(nneurons):
    neuron.append(p.Population(npop, cellclass=p.SpikeSourceArray))

nneuron=3
trial=20
DAT=DATA[4][nneuron][trial]
for direction in dirs:
    for n in range(len(neuron)):
	neuron[n].set('spike_times',DATA[direction][n][trial])

nout=2
out=[]
for i in range(nout):
    out.append(p.Population(npop, cellclass=p.IF_cond_exp))
    

initweight=0.08
prj=[]
for i in range(nneurons):
    for j in range(nout):
        prj.append(p.Projection(neuron[i], out[j], target="excitatory",method=p.AllToAllConnector()))
        prj[-1].setWeights(rand()*0.1)


for o in out:
    o.record()
    o.record_v()


for n in neuron:
  n.record()
  
p.run(2000)

outspikes=[]
outvolts=[]
for o in out:
    outspikes.append(o.getSpikes())
    outvolts.append(o.get_v())


    
      
fig = figure()
ax = fig.add_subplot(2,1,1)
hold(True)
for i in range(nout):
    ax.plot(outspikes[i][:,1],i*ones_like(outspikes[i][:,1]),'b|')
ax.set_ylim(-6,5)
for i in range(nneurons):
   
#    ax.plot(DATA[direction][i][trial],-1-i*ones_like(DATA[direction][i][trial]),'r|')
    inspikes=neuron[i].getSpikes()
    ax.plot(inspikes,-1-i*ones_like(inspikes),'r|')
ax2=fig.add_subplot(2,1,2)
ax2.plot(outvolts[0][:,1],outvolts[0][:,2])
    
