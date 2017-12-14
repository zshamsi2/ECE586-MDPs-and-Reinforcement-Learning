import matplotlib.pyplot as plt

import numpy as np

plt.rcParams.update({'font.size':18})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

totalTimes = np.loadtxt('totalTimes')
plt.scatter(np.arange(len(totalTimes)), totalTimes, color='orange', s=50)
plt.plot(totalTimes, '--', lw=2.5)

plt.xlabel('Epochs')
plt.ylabel('Number of steps to reach the final state')
plt.savefig('totalTime', dpi=300, bbox_inches='tight')

plt.show()
