import scipy.io 
import matplotlib.pyplot as plt
import numpy as np
import pickle

a1 = scipy.io.loadmat('../data/spontaneous_recording_nostruct.mat')

#exercise 1

unfilteredTrace = a1['I']
filteredTrace = a1['FI']
timeResolutionS = a1['TimeResolutionS']
samplingFrequency = a1['SamplingFrequencyHz'][0][0]

length = len(unfilteredTrace)
timePoints = (np.arange(0,length) * timeResolutionS)[0]

startPoint = 0
endPoint = 2 * samplingFrequency

plt.plot(timePoints[startPoint:endPoint], unfilteredTrace[startPoint:endPoint],
        color='0.7')
plt.plot(timePoints[startPoint:endPoint], filteredTrace[startPoint:endPoint],
'b')
plt.xlabel('Time/s')
plt.ylabel('Current/pA')

plt.suptitle("Filtered vs unfiltered trace, threshold crossings")
plt.savefig('../output/day2_figure5.png')

#exercise 3

#plt.plot(line)
threshold = 10

aboveThreshold  = filteredTrace>threshold
belowThreshold = filteredTrace<threshold
onset_idx = np.where(np.logical_and(belowThreshold[:-1], aboveThreshold[1:]))[0]
visualOnset_idx = np.where(np.logical_and(belowThreshold[startPoint:endPoint-1],
    aboveThreshold[startPoint+1:endPoint]))
visualOnset_idx *= timeResolutionS[0]
numCrossings = len(onset_idx)

line = plt.axhline(y=threshold, xmin=startPoint, xmax=endPoint, color='r')

visualOnset_idy = np.ones(len(visualOnset_idx)) * threshold
plt.plot(visualOnset_idx, visualOnset_idy, 'rD')

plt.draw()

#exercise 4

window = int(0.05 * samplingFrequency) 

PSCs=[unfilteredTrace[int(onset_idx[0]-window/2):int(onset_idx[0]+window/2)]]
for onsetTime in (onset_idx[1:]):
    PSCs = np.append(PSCs,[unfilteredTrace[int(onsetTime-window/2):int(onsetTime+window/2)]], axis=0)

#exercise 5

plt.figure()

timePoints = (np.arange(0,window) * timeResolutionS)[0]

for PSC in PSCs:
    plt.plot(timePoints, PSC, color='0.7')

averagePSC = []
stdPSC = []
for x in range(0, window):
    averagePSC.append(np.average(PSCs[:,x]))
    stdPSC.append(np.std(PSCs[:,x]))

plt.plot(timePoints, averagePSC, color='b')
plt.xlabel('Time/s')
plt.ylabel('Current/pA')
plt.suptitle("PSC traces, average PSC")
plt.savefig('../output/day2_figure6.png')

plt.figure()

upper = np.array(averagePSC) + np.array(stdPSC)
lower = np.array(averagePSC) - np.array(stdPSC)

plt.plot(timePoints, averagePSC, color='b')
plt.plot(timePoints, lower)
plt.plot(timePoints, upper)

plt.xlabel('Time/s')
plt.ylabel('Current/pA')
plt.suptitle("Average PSC +- standard deviation")

plt.draw()
plt.savefig('../output/day2_figure7.png')
maxAveragePeak = np.max(averagePSC)
print 'Max average peak: ', maxAveragePeak

#exercise 6

plt.figure()

peakList = []
for PSC in PSCs:
    peakList.append(np.max(PSC))

plt.xscale('log')
n, bins, patches = plt.hist(peakList, bins=10**np.linspace(1,2,20), normed=1, facecolor='green', alpha=0.75)

plt.xlabel('Peak current/pA')
plt.ylabel('Frequency')
plt.suptitle("Peak current frequency distribution")

plt.draw()
plt.savefig('../output/day2_figure8.png')
plt.show()
