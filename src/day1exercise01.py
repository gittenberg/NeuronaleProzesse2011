import scipy.io 
import matplotlib.pyplot as plt
import numpy as np
import pickle

a1 = scipy.io.loadmat('../data/A1_data_set_040528_boucsein_nostruct.mat')

#exercise 1

data = a1['V'][:,0]
interval = a1['SampleIntervalSeconds'][0][0]
length = len(data) * interval
#length of complete voltage trace = number of data points times interval =
#90.32 seconds
timePoints = np.arange(0,length, interval)

# 2 Sekunden entsprechen 2/(4e-05) = 50000 Datenpunkten
plot = plt.plot(timePoints[:50001], data[:50001])

ax = plt.gca()
plt.xlabel('Time/s')
plt.ylabel('Membrane potential/mV')
ax.set_ylim(-75, 20)
plt.draw()
#plt.show()

plt.savefig('../output/day1_figure1.png')

#exercise 2

theta = -10 #threshold value
dataSpikes  = data>theta
spikePoints = np.diff(dataSpikes)
spikeTimes = np.where(np.logical_and(spikePoints, dataSpikes[:-1]))[0]
numActionPotentials = len(spikeTimes)
avgFiringRate = numActionPotentials / length

#exercise 3

n = int(0.01 / interval)
spikeArray=[data[spikeTimes[0]-n:spikeTimes[0]+n]]
for spikeOnsetTime in (spikeTimes[1:-1]):
    spikeArray = np.append(spikeArray, [data[spikeOnsetTime-n:spikeOnsetTime+n]], axis=0)

averageSpike = []
varianceSpike = []
for x in range(0, 2*n):
    averageSpike.append(np.average(spikeArray[:,x]))
    varianceSpike.append(np.std(spikeArray[:,x]))

upper = np.array(averageSpike) + np.array(varianceSpike)
lower = np.array(averageSpike) - np.array(varianceSpike)

timePoints = [x * 0.04 for x in range(-250, 250)] 

plt.figure()
plt.xlabel('Time/ms')
plt.ylabel('Membrane potential/mV')
plt.plot(timePoints, averageSpike)
plt.plot(timePoints, lower)
plt.plot(timePoints, upper)
plt.plot(timePoints, np.array(varianceSpike))
plt.savefig('../output/day1_figure2.png')

#exercise 4

deleteList = []
for spikeOnsetTime in spikeTimes:
    for x in range(spikeOnsetTime-n, spikeOnsetTime+n):
	deleteList.append(x)

dataWithoutSpikes = np.delete(data, deleteList)

'''dataWithoutSpikes = data.copy()
for spikeOnsetTime in (spikeTimes)[::-1]:
    print spikeOnsetTime 
    dataWithoutSpikes = np.delete(dataWithoutSpikes, range(spikeOnsetTime+n,
        spikeOnsetTime-n, -1))

pickle.dump(dataWithoutSpikes, open("data.pkl", "wb"))

#dataWithoutSpikes = pickle.load(open("data.pkl"))
'''

average = np.average(dataWithoutSpikes)
std = np.std(dataWithoutSpikes)

plt.figure()
plt.xlabel('Membrane potential/mV')
plt.ylabel('Probability density')
n, bins, patches = plt.hist(dataWithoutSpikes, 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
plt.savefig('../output/day1_figure3.png')
