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

plt.draw()
plt.savefig('../output/day3_figure2.png')

plt.figure()

plt.bar(bins[1:], np.array(n) / float(sum(n)) ** 2 / 0.005, width=0.005)  

plt.draw()
plt.savefig('../output/day3_figure3.png')

plt.show()
