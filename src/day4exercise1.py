from pylab import *
import pyNN.neuron as p
import time

p.setup(timestep=0.01, quit_on_end=False)

def exercise1():
    hugs = p.Population(1, cellclass=p.HH_cond_exp)
    dcsource = p.DCSource(amplitude=1., start=100., stop=600)
    dcsource.inject_into(hugs)

    hugs.record()
    hugs.record_v()

    p.run(1000.)

    pot = hugs.get_v()
    spikes = hugs.getSpikes()

    plot(pot[:,1], pot[:,2])
    plot(spikes[:,1], [-40]*len(spikes), '.', color='r')
    line = axhline(y=-40, xmin=0, xmax=len(pot[:,1]), color='r')
    xlabel('Time/ms')
    ylabel('Current/mV')
    savefig('../output/day4_figure1.png')
    show()

def exercise2():
  hugs = p.Population(1, cellclass=p.HH_cond_exp)
  start = 100.
  stop = 1100.
  frequency = []
  currentSpace = linspace(0.1,10,100)
  for current in currentSpace:
    hugs.record()
    hugs.record_v()
    dcsource = p.DCSource(amplitude=current, start=start, stop=stop)
    dcsource.inject_into(hugs)
    p.run(1100.)
    spikes = hugs.getSpikes()
    frequency.append(len(spikes) / (stop - start))
    p.reset() 
  plot(currentSpace, frequency)
  xlabel('Current/nA')
  ylabel('Frequency/kHz')
  savefig('../output/day4_figure2.png')
  show()
    
def exercise3():
  p.setup(quit_on_end=False, timestep=0.01)
  iandf = p.Population(1, cellclass=p.IF_cond_exp)
  dcsource = p.DCSource(amplitude=1., start=100., stop=1000.)
  dcsource.inject_into(iandf)
  iandf.record()
  iandf.record_v()
  p.run(1200.)
  pot = iandf.get_v()
  spikes = iandf.getSpikes()
  plot(pot[:,1], pot[:,2], color='b')
  plot(spikes[:,1], [-60]*len(spikes), '.', color='r')
  line = axhline(y=-60, xmin=0, xmax=len(pot[:,1]), color='r')
  xlabel('Time/ms')
  ylabel('Current/mV')
  savefig('../output/day4_figure3.png')
  show()
  
def exercise4_hugs():
    startTime = time.clock()
    hugs = p.Population(1, cellclass=p.HH_cond_exp)
    dcsource = p.DCSource(amplitude=0.15, start=100., stop=100000)
    dcsource.inject_into(hugs)
    p.run(100000.)
    print(time.clock() - startTime)
    
def exercise4_iandf():
    startTime = time.clock()
    hugs = p.Population(1, cellclass=p.IF_cond_exp)
    dcsource = p.DCSource(amplitude=0.9, start=100., stop=100000)
    dcsource.inject_into(hugs)
    p.run(100000.)
    print(time.clock() - startTime)
    
def exercise4_curve(tau_refrac):
  hugs = p.Population(1, cellclass=p.IF_cond_exp)
  hugs.set("tau_refrac", tau_refrac)
  start = 100.
  stop = 1100.
  frequency = []
  currentSpace = linspace(0.1,5.0,50)
  for current in currentSpace:
    hugs.record()
    hugs.record_v()
    dcsource = p.DCSource(amplitude=current, start=start, stop=stop)
    dcsource.inject_into(hugs)
    p.run(1100.)
    spikes = hugs.getSpikes()
    frequency.append(len(spikes) / (stop - start))
    print(current, frequency[-1])
    p.reset() 
  return currentSpace, frequency

def exercise4_iandfCurves():
  curveCurrent, curveFreq = exercise4_curve(0.0)
  plt1 = plot(curveCurrent, curveFreq)
  curveCurrent, curveFreq = exercise4_curve(2.0)
  plt2 = plot(curveCurrent, curveFreq)
  legend((plt1, plt2), ('Post-spike refractory time = 0ms', 'Post-spike refractory time = 2ms'))
  xlabel('Current/nA')
  ylabel('Frequency/kHz')
  savefig('../output/day4_figure4.png')
  show()
  
def exercise5(tau_m, v_thresh=-55.):
  hugs = p.Population(1, cellclass=p.IF_cond_exp)
  hugs.set("tau_refrac", 2.0)
  hugs.set("tau_m", tau_m)
  hugs.set("v_thresh", v_thresh)
  
  
  start = 100.
  stop = 400.
  
  dcsource = p.DCSource(amplitude=0.95, start=start, stop=stop)
  dcsource.inject_into(hugs)

  hugs.record()
  hugs.record_v()

  p.run(500.)

  pot = hugs.get_v()
  spikes = hugs.getSpikes()
  
  p.reset() 
 
  return pot, spikes, (stop - start)
    
def exercise5_curves():
  pot, spikes, dummy = exercise5(20)
  plt1 = plot(pot[:,1], pot[:,2], color='b')
  plot(spikes[:,1], [-50]*len(spikes), 'D', color='b')
  pot, spikes, dummy = exercise5(40)
  plt2 = plot(pot[:,1], pot[:,2], color='r')
  plot(spikes[:,1], [-50]*len(spikes), 'D', color='r')
  legend((plt1, plt2), ('Time constant of the membrane = 20ms', 'Time constant of the membrane = 40ms'))
  title('Effect of tau_m on firing rate 1/2')
  xlabel('Time/ms')
  ylabel('Current/mV')
  savefig('../output/day4_figure5.png')
  show()
  
def exercise5_varTimeConstant():
  timeConstants = linspace(10, 100, 91)
  spikeFrequencies = []
  for tau in timeConstants:
    spikeFrequencies.append(len(exercise5(tau)[1])/exercise5(tau)[2])
    print('.')
  plot(timeConstants, spikeFrequencies)
  title('Effect of tau_m on firing rate 2/2')
  xlabel('tau_m/ms')
  ylabel('Firing rate/kHz')
  savefig('../output/day4_figure6.png')
  show()
  
def exercise5_varThreshold():
  thresholds = linspace(-55, -45, 11)
  spikeFrequencies = []
  for th in thresholds:
    pot, spikes, time = exercise5(7, th)
    spikeFrequencies.append(len(spikes)/time)
    print('.')
  plot(thresholds, spikeFrequencies)
  title('Effect of threshold voltage on firing rate')
  xlabel('Threshold Voltage/mV')
  ylabel('Firing rate/kHz')
  savefig('../output/day4_figure7.png')
  show()
  

  
  
  
  
  
#if (__name__ == '__main__'): exercise5_curves()
     
        
