import pyNN.neuron as p
import numpy
p.setup(timestep=0.1)
from pylab import *
import NeuroTools.stgen as nts
import pickle

timeShift = 50

stg = nts.StGen()
st1 = stg.poisson_generator(rate=10., t_start=0., t_stop=20000.,array=True)
st2 = stg.poisson_generator(rate=10., t_start=0., t_stop=20000.,array=True)
st3 = numpy.union1d(st1-timeShift, st2+timeShift)
st3 = st3[st3>0]
ss1 = p.Population(1, cellclass=p.SpikeSourceArray,
        cellparams={'spike_times':st1})
ss2 = p.Population(1, cellclass=p.SpikeSourceArray, cellparams={'spike_times':st2})
ss3 = p.Population(1, cellclass=p.SpikeSourceArray, cellparams={'spike_times':st3})
tn = p.Population(1, cellclass=p.IF_cond_exp)
tn.record()
prj3 = p.Projection(ss3, tn, method=p.AllToAllConnector())
prj3.setWeights(0.08)
syndyn = p.SynapseDynamics(fast=None,
     slow=p.STDPMechanism(
            timing_dependence=p.SpikePairRule(tau_plus=10.,
                                              tau_minus=10.),
            weight_dependence=p.AdditiveWeightDependence(w_min=0.01,
w_max=0.08,
                                                         A_plus=0.003,
                                                         A_minus=0.005)))
prj1 = p.Projection(ss1,tn, 
                    method=p.AllToAllConnector(),
                    synapse_dynamics=syndyn)
prj2 = p.Projection(ss2,tn,  
                    method=p.AllToAllConnector(),
                    synapse_dynamics=syndyn)
weight = 0.04
prj1.setWeights(weight)
prj2.setWeights(weight)
times = numpy.ones(2000)*10. # 2000 * 10 ms = 20000 ms
prj1_weights = numpy.zeros_like(times)
prj2_weights = numpy.zeros_like(times)
for i,t in enumerate(times):
        p.run(t)
        prj1_weights[i] = prj1.getWeights()[0]
        prj2_weights[i] = prj2.getWeights()[0]

spikes = tn.getSpikes()

figure()
plot(numpy.cumsum(times), prj1_weights, label='prj1', color='blue')
plot(numpy.cumsum(times), prj2_weights, label='prj2', color='red')
plot(spikes, [0.04]*len(spikes), '|', color='black', markersize=10.0)
plot(st1, [0.042]*len(st1), '|', color='blue', markersize=10.0)
plot(st2, [0.042]*len(st2), '|', color='red', markersize=10.0)
ax = gca()
ax.set_ylabel('weight [nA]')
ax.set_xlabel('time [ms]')
legend()

savefig('../output/day5figure8.png')

draw()
show()
