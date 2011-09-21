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

