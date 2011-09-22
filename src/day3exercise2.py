import scipy.io 
import matplotlib.pyplot as plt
import numpy as np
import pickle

pp1 = scipy.io.loadmat('../data/PP1_Gamma_course_day2_nostruct.mat')

spikeTimes = pp1['Spikes'][0][2][0]


timeDifs = []
for spikeTime1 in spikeTimes:
    list = []
    for spikeTime2 in spikeTimes:
        dif = spikeTime1 - spikeTime2
        if not dif == 0 and not dif > 0.5 and not dif < 0:
            list.append(dif)
    timeDifs.append(np.array(list))

pickle.dump(timeDifs, open("timeDifs.pkl", "wb"))


#timeDifs = pickle.load(open("timeDifs.pkl"))

flatTimeDifs = [timeDif for sublist in timeDifs for timeDif in sublist]

n, bins, patches = plt.hist(flatTimeDifs, 100, normed=0)
plt.xlabel('Time/s')

plt.figure()

plt.bar(bins[1:], np.array(n) / float(sum(n)) ** 2 / 0.005, width=0.005)  

plt.draw()
plt.show()

'''file = open('../data/SFA_reg_1.2q_200tau_mn.gdf')

s = []
id = []
trialSpikeGroups = {}
    
for line in file:
    list = line.strip().split('  ')
    id.append(int(float((list[0]))))
    s.append(float(list[1]) / 10.0 - 3000.0)

for i in id:
    if not trialSpikeGroups.has_key(i):
        trialSpikeGroups[i] = []

for i, time in zip(id, s):
    trialSpikeGroups[i].append(time)

file.close()

print 'Number of spikes: ', len(id)

plt.figure()
plt.subplot(3,1,1)
plt.ylabel('Trial')
plt.xlabel('Time/ms')
plt.plot(s, id, '.', markersize=1)

#exercise 4

binWidth = 20
binArray = np.linspace(-3000, 3000, 3000/binWidth)
h, bins = np.histogram(s, binArray)

#exercise 5

plt.subplot(3,1,2)
plt.ylabel('Rate 1/s')
plt.xlabel('Time/ms')
b = plt.bar(bins[0:h.size], h)
plt.setp(b, width=40)
plt.legend((b[0],), ('bins = 20ms',), loc='upper left')


plt.subplot(3,1,3)
plt.ylabel('Rate 1/s')
plt.xlabel('Time/ms')
binWidth = 50
binArray = np.linspace(-3000, 3000, 3000/binWidth)
h, bins = np.histogram(s, binArray)
b = plt.bar(bins[0:h.size], h)
plt.setp(b, width=100)
plt.legend((b[0],), ('bins = 50ms',), loc='upper left')

#exercise 6

stimulusSpikeGroups = [[entry for entry in trialSpikeGroups[trial] if entry >=
    0 and entry <= 2000] for trial in trialSpikeGroups]

spikeCount = [len(trial) for trial in stimulusSpikeGroups]
FF = np.std(spikeCount)**2 / np.mean(spikeCount)

print 'Fano factor:', FF

plt.savefig('../output/day3_figure1.png')

plt.draw()
plt.show()
'''

'''
a1 = scipy.io.loadmat('../data/synaptic_response_variability_nostruct.mat')

#exercise 1

curTrc = a1['I'][:,0]
shtCmdVoltage = a1['Z'][:,0]
tmpCurRes = a1['I_TimeResolutionS'][0][0]
tmpShtRes = a1['Z_TimeResolutionS'][0][0]

stimOnsetPoints = np.where(np.diff(shtCmdVoltage) < -3.5)[0]
stimOnsetTimes = stimOnsetPoints * tmpShtRes
stimOnsetGroup = np.reshape(stimOnsetPoints, (3,len(stimOnsetPoints)/3), order='F')

#exercise 2

targetGroup = []
for group in stimOnsetGroup:
	startPoint = int(group[0]*tmpShtRes/tmpCurRes)
	endPoint = int((group[0]*tmpShtRes + 0.1)/tmpCurRes)
	EPSCgroup = [np.array(curTrc[startPoint:endPoint][:2083])]
	for EPSC in group[1:]:
	        startPoint = int(EPSC*tmpShtRes/tmpCurRes)
        	endPoint = int((EPSC*tmpShtRes + 0.1)/tmpCurRes)
        	EPSCgroup = np.append(EPSCgroup, [np.array(curTrc[startPoint:endPoint][:2083])], axis=0)
	targetGroup.append(EPSCgroup)

avEPSCgroup = []
for EPSCgroup in targetGroup:
	avEPSCgroup.append([np.average(EPSCgroup[:,x]) for x in range(0, len(EPSCgroup[0]))])

#exercise 3

peakGroup = []
meanGroup = []
stdGroup = []
varCofGroup = []
for EPSCgroup in targetGroup:
	peakValues = []
	for EPSC in EPSCgroup:
		peakValues.append(np.max(EPSC))
	peakGroup.append(peakValues)
	meanGroup.append(np.mean(peakValues))
	stdGroup.append(np.std(peakValues))
	varCofGroup.append(np.std(peakValues / np.mean(peakValues)))
	
timePoints = np.arange(0, 2083) * tmpCurRes

fig = plt.figure()
plt.suptitle('Average excitatory postsynaptic currents')
plt.subplot(3,1,1)
for EPSC in targetGroup[0]:
	plt.plot(timePoints, EPSC, color='0.7')
plt.plot(timePoints, avEPSCgroup[0])
plt.text(0.06, 120, 'mean='+str(meanGroup[0]))
plt.text(0.06, 100, 'std='+str(stdGroup[0]))
plt.text(0.06, 80, 'cof. var.='+str(varCofGroup[0]))
plt.ylabel('Current/pA')
plt.subplot(3,1,2)
for EPSC in targetGroup[1]:
	plt.plot(timePoints, EPSC, color='0.7')
plt.plot(timePoints, avEPSCgroup[1])
plt.text(0.06, 60, 'mean='+str(meanGroup[1]))
plt.text(0.06, 50, 'std='+str(stdGroup[1]))
plt.text(0.06, 40, 'cof. var.='+str(varCofGroup[1]))
plt.ylabel('Current/pA')
plt.subplot(3,1,3)
for EPSC in targetGroup[2]:
	plt.plot(timePoints, EPSC, color='0.7')
plt.plot(timePoints, avEPSCgroup[2])
plt.text(0.06, 120, 'mean='+str(meanGroup[2]))
plt.text(0.06, 100, 'std='+str(stdGroup[2]))
plt.text(0.06, 80, 'cof. var.='+str(varCofGroup[2]))
plt.ylabel('Current/pA')
plt.xlabel('Time/s')

plt.savefig('../output/day2_figure9.png')

plt.draw()
plt.show()
'''

