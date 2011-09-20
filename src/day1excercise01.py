import scipy.io 
import matplotlib.pyplot as plt
import numpy as np

a1 = scipy.io.loadmat('A1_data_set_040528_boucsein_nostruct.mat')

#excercise 1

# 2 Sekunden entsprechen 2/(4e-05) = 50000 Datenpunkten
data = a1['V'][:,0]
interval = a1['SampleIntervalSeconds'][0][0]
length = len(data) * interval
#length of complete voltage trace = number of data points times interval =
#90.32 seconds
timePoints = np.arange(0,length, interval)

plot = plt.plot(timePoints[:50001], data[:50001])

ax = plt.gca()
plt.xlabel('Time/s')
plt.ylabel('Membrane potential/mV')
ax.set_ylim(-75, 20)
plt.draw()
#plt.show()

plt.savefig('figure1.png')

#excercise 2

theta = -20 #threshold value
dataSpikes  = data>theta
spikePoints = np.diff(dataSpikes)
spikeTimes = np.where(np.logical_and(spikePoints, dataSpikes[:-1]))[0]
numActionPotentials = len(spikeTimes)
avgFiringRate = numActionPotentials / length
