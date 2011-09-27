import pyNN.neuron as p
import numpy
from pylab import *
import NeuroTools.stgen as nts
import pickle

timestep = 0.1
duration = 1000.
p.setup(timestep=timestep)

def LatInhibCircuit(weights = 0.5):
    inp1Num = 10
    inp2Num = 10
    inh1Num = 10
    inh2Num = 10

    inh1 = p.Population(inh1Num, cellclass=p.IF_cond_exp)
    inh2 = p.Population(inh2Num, cellclass=p.IF_cond_exp)
    inp1 = p.Population(inp1Num, cellclass=p.IF_cond_exp)
    inp2 = p.Population(inp2Num, cellclass=p.IF_cond_exp)
    p1 = p.Population(inp1Num, cellclass=p.SpikeSourcePoisson)
    p2 = p.Population(inp2Num, cellclass=p.SpikeSourcePoisson)
    rate1 = 300./inp1Num
    rate2 = 600./inp2Num
    p1.set('rate', rate1)
    p2.set('rate', rate2)

    p1ToInp1 = p.Projection(p1, inp1, target='excitatory', method=p.OneToOneConnector())
    p2ToInp2 = p.Projection(p2, inp2, target='excitatory', method=p.OneToOneConnector())
    inp1toInh1 = p.Projection(inp1, inh1, target='excitatory', method=p.OneToOneConnector())
    inp2toInh2 = p.Projection(inp2, inh2, target='excitatory', method=p.OneToOneConnector())
    inh1ToInp2 = p.Projection(inh1, inp2, target='inhibitory', method=p.OneToOneConnector())
    inh2ToInp1 = p.Projection(inh2, inp1, target='inhibitory', method=p.OneToOneConnector())

    p1ToInp1.setWeights(0.05)
    p2ToInp2.setWeights(0.05)
    inp1toInh1.setWeights(weights)
    inp2toInh2.setWeights(weights)
    inh1ToInp2.setWeights(0.5)
    inh2ToInp1.setWeights(0.5)

    p1.record()
    p2.record()
    inp1.record()
    inp2.record()
    inh1.record()
    inh2.record()
    inh1.record_v()
    inh2.record_v()
    inp1.record_v()
    inp2.record_v()

    p.run(duration)

    inp1V = inp1.get_v()
    inp1Vtmp = [np.array(inp1V[inp1V[:,0] == 0][:,2])]
    for index in range(1,inp1Num):
        inp1Vtmp = np.append(inp1Vtmp, [inp1V[inp1V[:,0] == index][:,2]], axis=0)
    inp1Av = np.mean(inp1Vtmp, 0)

    inp2V = inp2.get_v()
    inp2Vtmp = [np.array(inp2V[inp2V[:,0] == 0][:,2])]
    for index in range(1,inp2Num):
        inp2Vtmp = np.append(inp2Vtmp, [inp2V[inp2V[:,0] == index][:,2]], axis=0)
    inp2Av = np.mean(inp2Vtmp, 0)

    inh1V = inh1.get_v()
    inh1Vtmp = [np.array(inh1V[inh1V[:,0] == 0][:,2])]
    for index in range(1,inh1Num):
        inh1Vtmp = np.append(inh1Vtmp, [inh1V[inh1V[:,0] == index][:,2]], axis=0)
    inh1Av = np.mean(inh1Vtmp, 0)

    inh2V = inh2.get_v()
    inh2Vtmp = [np.array(inh2V[inh2V[:,0] == 0][:,2])]
    for index in range(1,inh2Num):
        inh2Vtmp = np.append(inh2Vtmp, [inh2V[inh2V[:,0] == index][:,2]], axis=0)
    inh2Av = np.mean(inh2Vtmp, 0)

    figure()
    subplot(2,2,1)
    plot(inp1Av)
    plot(p1.getSpikes()[:,1] / timestep, [max(inp1Av)]*len(p1.getSpikes()), '|', markersize=20.0)

    subplot(2,2,2)
    plot(inp2Av)
    plot(p2.getSpikes()[:,1] / timestep, [max(inp2Av)]*len(p2.getSpikes()), '|', markersize=20.0)

    subplot(2,2,3)
    plot(inh1Av)

    subplot(2,2,4)
    plot(inh2Av)
    

    draw()
    show()

    numSpikes1 = len(inp1.getSpikes())
    numSpikes2 = len(inp2.getSpikes())
    rateOut1 = numSpikes1/duration
    rateOut2 = numSpikes2/duration

    kappaIn = (rate1-rate2)/(rate1+rate2)
    kappaOut = (rateOut1-rateOut2)/(rateOut1+rateOut2)

    p.reset()

    return abs(kappaIn), abs(kappaOut)

for i in arange(0.1, 1.0, 0.1):
    print i
    print LatInhibCircuit(i)

