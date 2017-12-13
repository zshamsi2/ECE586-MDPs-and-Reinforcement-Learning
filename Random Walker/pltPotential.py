import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':18})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

#trjs_theta = np.load('Trjs.npy')

#x = np.array(trjs_theta[0])
#y = np.array(trjs_theta[1])
# parameters in Mueller potential

aa = [-2, -20, -20, -20, -20] # inverse radius in x
bb = [0, 0, 0, 0, 0] # radius in xy
cc = [-20, -20, -2, -20, -20] # inverse radius in y
AA = 30*[-200, -120, -200, -80, -80] # strength

#XX = [1, 0, 0, 0, 0.4] # center_x
#YY = [0, 0, 1, 0.4, 0] # center_y
XX = [1.3, 0.3, 0.3, 0.3, 0.7] 
YY = [0.35, 0.35, 1.35, 0.75, 0.35]

zxx = np.mgrid[-1:2.51:0.01]
zyy = np.mgrid[-1:2.51:0.01]
xx, yy = np.meshgrid(zxx, zyy)


V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
for j in range(1,5):
        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))

fig = plt.figure()
ax = fig.add_subplot(111)
                
ax.contourf(xx,yy,np.minimum(V1,200), 40)

plt.xlabel('x')
plt.ylabel('y')
plt.ylim([0, 2])
plt.xlim([0, 2])


#plt.grid(linewidth=2)

#plt.plot(x, y, 'o', color='white', alpha=0.2, mec="black")
plt.savefig('L-potential.png', dpi=100)
