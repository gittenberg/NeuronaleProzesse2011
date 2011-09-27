from pylab import *
import pyNN.neuron as p
import time

timestep = 0.1
p.setup(timestep, quit_on_end=False)

def exercise1(n_inhib=25, n_excit=100):
    sspe = p.Population(n_excit, cellclass=p.SpikeSourcePoisson)
    sspi = p.Population(n_inhib, cellclass=p.SpikeSourcePoisson)
    sspe.set('rate', 10.)
    sspi.set('rate', 10.)
    sspe.record()
    sspi.record()

    n = p.Population(1, cellclass=p.IF_cond_exp)
    n.record()
    n.record_v()
    n.record_gsyn()

    prje = p.Projection(sspe, n, target="excitatory", method=p.AllToAllConnector())
    prje.setWeights(0.0001)
    prji = p.Projection(sspi, n, target="inhibitory", method=p.AllToAllConnector())
    prji.setWeights(0.0001)

    p.run(1000.)
    pot = n.get_v()
    spikes_e = sspe.getSpikes()
    spikes_i = sspi.getSpikes()

    subplot(2,1,1)
    ylabel('Membrane potential/mV')
    plot(pot[:,1], pot[:,2])

    subplot(2,1,2)

    for spike in spikes_i:
        plt.plot(spike[1], spike[0], '.', markersize=1, color='g')
    for spike in spikes_e:
        plt.plot(spike[1], spike[0]+n_inhib, '.', markersize=1, color='r')

    xlabel('Time/ms')
    ylabel('Source ID')

    ylim(0, n_inhib + n_excit)

    savefig('../output/day5figure3.png')

    '''

    figure()

    window = 1000
    STAgroup = []
    for spike in spikes_i:
        startTime = int(spike[1]/timestep)
        if startTime > window and len(pot) > startTime + 2*window:
            STAgroup.append(pot[startTime-window:startTime+2*window,2])
    
    import pdb; pdb.set_trace()
    timePoints = np.arange(0,3*window)


    for STA in STAgroup:
        if len(timePoints) == len(STA):
            plt.plot(timePoints, STA[:,2], color='0.7')

    valueGroup = [np.array(STAgroup[1][:,2])]
    for STA in STAgroup[2:]:
        if len(timePoints) == len(STA):
            valueGroup = np.append(valueGroup, [np.array(STA[:,2])], axis=0)


    averageSTA = []
    for x in range(0, 3*window):
        averageSTA.append(np.average(np.array(valueGroup[:,0])))
    
    
    plot(timePoints, averageSTA)
    '''
    figure()
    xlabel('Time/ms')
    ylabel(u'Inhibitory conductance/\u00B5S')

    gsyn = n.get_gsyn()
    plot(gsyn[:,1], gsyn[:,2], color='r')
    plot(gsyn[:,1], gsyn[:,3], color='g')

    savefig('../output/day5figure4.png')

    draw()
    show()

'''
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
'''
