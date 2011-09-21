import scipy.io 
import matplotlib.pyplot as plt
import numpy as np
import pickle

a1 = scipy.io.loadmat('../data/synaptic_response_variability_nostruct.mat')

#exercise 1

curTrc = a1['I'][:,0]
shtCmdVoltage = a1['Z'][:,0]
tmpCurRes = a1['I_TimeResolutionS'][0][0]
tmpShtRes = a1['Z_TimeResolutionS'][0][0]

openingPoints = np.where(np.diff(shtCmdVoltage) > 4)[0] * tmpShtRes
openingGroup = np.reshape(openingPoints, (3,len(openingPoints)/3), order='F')

#exercise 2

EPSCs = []
for group in openingGroup:
    startTime = int(group[0]/tmpCurRes)
    endTime = int(group[0]/tmpCurRes +0.1/tmpCurRes)
    EPSCs0 = [np.array(curTrc[startTime:endTime])]
    for point in group[1:]:
        startTime = int(point/tmpCurRes)
        endTime = int(point/tmpCurRes + 0.1/tmpCurRes)
        EPSCs0 = np.append(EPSCs0, [np.array(curTrc[startTime:endTime])],
                axis=0)
    EPSCs.append(EPSCs0)

averageEPSCs = []
for group in EPSCs:
    averagePSC = np.array(np.average(np.array(group)[:,0]))
    for x in range(1, len(group[0])):
        averagePSC = np.append(averagePSC, np.average(np.array(group)[:,x]))
    averageEPSCs.append(averagePSC)

pickle.dump(averageEPSCs, open('averageEPSCs.pkl', 'wb'))

#averageEPSCs = pickle.load(open('averageEPSCs.pkl'))

timePoints = np.arange(0, 2083) * tmpCurRes

fig = plt.figure()
plt.subplot(3,1,1)
plt.plot(timePoints, averageEPSCs[0])
plt.subplot(3,1,2)
plt.plot(timePoints, averageEPSCs[1])
plt.subplot(3,1,3)
plt.plot(timePoints, averageEPSCs[2])

plt.xlabel('Time/s')
plt.ylabel('Current/pA')

plt.draw()
plt.show()
'''unfilteredTrace = a1['I']
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
'''
