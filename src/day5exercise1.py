from pylab import *
import pyNN.neuron as p
import time

p.setup(timestep=0.1, quit_on_end=False)

def exercise1(v_rest = 65.0):
    ssp = p.Population(1, cellclass=p.SpikeSourcePoisson)
    ssp.set('rate', 30.)
    ssp.record()
    n = p.Population(1, cellclass=p.IF_cond_exp)

    n.record_v()
    n.record()
    n.record_gsyn()
    n.set('v_rest', v_rest) 
    prj = p.Projection(ssp, n, target="excitatory", method=p.AllToAllConnector())
    prj.setWeights(0.001)
    p.run(1000)
    gsyn = n.get_gsyn()
    pot = n.get_v()
    spikes = ssp.getSpikes()
    plot(pot[:,1], pot[:,2])
    plot(spikes[:,1], [max(pot[:,2])]*len(spikes), '|', color='r', markersize=20.0)

    xlabel('Time/ms')
    ylabel('Membrane potential/mV')
    title('EPSPs after excitatory postsynaptic signal')

    plot(gsyn[:,1], gsyn[:,2])
    plot(spikes[:,1], [0]*len(spikes), '|', color='r', markersize=20.0)

    xlabel('Time/ms')
    ylabel('Membrane potential/mV')
    title('Excitatory conductance')

    savefig('../output/day5figure2.png')

    return pot, gsyn, spikes


def exercise2(v_rest = -65.0):
    ssp = p.Population(1, cellclass=p.SpikeSourcePoisson)
    ssp.set('rate', 30.)
    
    ssp.record()
    
    n = p.Population(1, cellclass=p.IF_cond_exp)
    
    n.set('v_rest', v_rest)
    n.record_v()
    n.record()
    n.record_gsyn()

    prj = p.Projection(ssp, n, target="excitatory", method=p.AllToAllConnector())
    prj.setWeights(0.000)

    sspi = p.Population(1, cellclass=p.SpikeSourcePoisson)
    sspi.set('rate', 30.)
    sspi.record()

    prji = p.Projection(sspi, n, target='inhibitory',
            method=p.AllToAllConnector())

    prji.setWeights(0.005)
    gsyn = n.get_gsyn()
    p.run(1000)
    gsyn = n.get_gsyn()
    pot = n.get_v()
    spikes = sspi.getSpikes()

    plot(pot[:,1], pot[:,2])
    #plot(gsyn[:,1], gsyn[:,2])
    plot(spikes[:,1], [max(pot[:,2])]*len(spikes), '|', color='r', markersize=20.0)

    xlabel('Time/ms')
    ylabel('Membrane potential/mV')
    title('IPSPs after inhibitory presynaptic signal')

    savefig('../output/day5figure3.png')

    return pot, gsyn, spikes


def subplot(v_rest = -65.0, out = '../output/day5figure1.png'):
    figure(figsize=(12, 10))

    suptitle('Threshold: ' + str(v_rest) +'mV')

    epspPot, econdGsyn, epspSpikes = exercise1(v_rest)
    p.reset()
    p.reset()
    ipspPot, icondGsyn, ipspSpikes = exercise2(v_rest)

    plt.subplot(2,2,1)
    plot(epspPot[:,1], epspPot[:,2])
    plot(epspSpikes[:,1], [max(epspPot[:,2])]*len(epspSpikes), '|', color='r', markersize=20.0)
    xlabel('Time/ms')
    ylabel('Membrane potential/mV')
    title('EPSPs after excitatory postsynaptic signal')

    plt.subplot(2,2,2)
    plot(ipspPot[:,1], ipspPot[:,2])
    plot(ipspSpikes[:,1], [max(ipspPot[:,2])]*len(ipspSpikes), '|', color='r', markersize=20.0)
    xlabel('Time/ms')
    ylabel('Membrane potential/mV')
    title('IPSPs after inhibitory presynaptic signal')

    plt.subplot(2,2,3)
    plot(econdGsyn[:,1], econdGsyn[:,2])
    plot(epspSpikes[:,1], [max(econdGsyn[:,2])]*len(epspSpikes), '|', color='r', markersize=20.0)
    xlabel('Time/ms')
    ylabel(u'Excitatory conductance/\u00B5S')

    plt.subplot(2,2,4)
    plot(icondGsyn[:,1], icondGsyn[:,3])
    plot(ipspSpikes[:,1], [max(icondGsyn[:,3])]*len(ipspSpikes), '|', color='r',markersize=20.0)
    xlabel('Time/ms')
    ylabel(u'Inhibitory conductance/\u00B5S')

    savefig(out)

    draw()
    show()

def subplot2():
    subplot(v_rest=-100.0, out='../output/day5figure2.png')

