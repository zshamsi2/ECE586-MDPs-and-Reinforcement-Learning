# Random Walker Simulation Environment
class walkerSimulation:

        ## public
        def __init__(self):
                self.Q = {}
                self.r = 1#number of rounds
                self.s = 1# length of simulations
                self.N = 1# number of parallel simulations

        ## private
             
        def updateState(self, trj): ## Need Change
                # map coordinate space to reaction coorinates space
                import numpy as np
                s = []
                for i in range(len(trj[0])):
                        x = trj[0][i]
                        y = trj[1][i]
                        if x>1.5:
                                if y<0.5:
                                        s.append(1)
                                elif y<1:
                                        s.append(5)
                                elif y<1.5:
                                        s.append(9)                                        
                                else:
                                        s.append(13)
                        elif x>1:
                                if y<0.5:
                                        s.append(2)
                                elif y<1:
                                        s.append(6)
                                elif y<1.5:
                                        s.append(10)                                        
                                else:
                                        s.append(14) 
                        elif x>0.5:
                                if y<0.5:
                                        s.append(3)
                                elif y<1:
                                        s.append(7)
                                elif y<1.5:
                                        s.append(10)                                        
                                else:
                                        s.append(15)
                        else:
                                if y<0.5:
                                        s.append(4)
                                elif y<1:
                                        s.append(8)
                                elif y<1.5:
                                        s.append(11)                                        
                                else:
                                        s.append(16)  
                                        
                s = np.unique(s)
                firstFlag = True
                        
                for i in s:
                        if firstFlag:
                                s_1=str(i)
                                firstFlag = False
                        else:
                                s_1=s_1+'0'+str(i)
                return s_1
        
        def findU(self, j, k, method='QL'):
                import numpy as np
                P_k = 90/(100+float(k))
                P_k = 90/(100+float(k))
                P_k = 0.95
                random = np.random.random()
                if random<P_k: # minimize Q function
                                ###################### minimization over v
                                iu = np.array(list(self.Q.keys()))
                                indexes = np.where(iu==j)
                                jus = []
                                firstFlag = True
                                for ii in range(len(indexes[0])):
                                        if indexes[1][ii]==0:  # if i is equal to j, not u
                                                i_index = indexes[0][ii] # first index of indexes
                                                iu_z = iu[i_index]
                                                #jus.append(iu[i_index])
                                                Q_val = self.Q[int(iu_z[0]), int(iu_z[1])]['Q']
                                                if firstFlag:
                                                        Q_min = Q_val
                                                        u_min = iu_z[1]
                                                        firstFlag=Fales
                                                else:
                                                        if Q_val<Q_min:
                                                                Q_min = Q_val
                                                                u_min = iu_z[1]
                                if firstFlag:
                                        Q_min=0
                                        u = np.random.choice([1, 2, 3, 4])
                        
                else:
                        u = np.random.choice([1, 2, 3, 4])
                        #u = np.random.choice(['u','d','r','l'])
                return u
                
                
                
        def isFinal(trjs_y):
                import numpy as np
                if np.any(trjs_y>1.5):
                        return True
                return False

        def cost(self, i, j): # change
                if i==j:
                        cost = 0
                else:
                        cost = -1 
                return cost

        def updateQ(self, i, j, u, cost):
                import numpy as np
                i = int(i)
                u = int(u)
                # Q format is: (i,u)=(Q_value, k)
                try:
                        k = self.Q[i,u]['k']
                        Q_old = self.Q[i,u]['Q']
                except:
                        k = 0
                        Q_old = 0
                epsilon_k = 90./(100+k)
                epsilon_k = 90./(100+k)
                alpha = 1
                
                ###################### minimization over v
                iu = np.array(list(self.Q.keys()))
                indexes = np.where(iu==j)
                jus = []
                firstFlag = True
                for ii in range(len(indexes[0])):
                        if indexes[1][ii]==0:  # if i is equal to j, not u
                                i_index = indexes[0][ii] # first index of indexes
                                iu_z = iu[i_index]
                                #jus.append(iu[i_index])
                                Q_val = self.Q[iu_z[0], iu_z[1]]['Q']
                                if firstFlag:
                                        Q_min = Q_val
                                        firstFlag=Fales
                                else:
                                        if Q_val<Q_min:
                                                Q_min = Q_val
                if firstFlag:
                        Q_min=0
                ############################################
                c = cost
                if k==0:
                        Q_new = (c+alpha*Q_min)
                        self.Q[i,u] ={'k':1, 'Q':Q_new}
                else:
                        Q_new = (1-epsilon_k)*Q_old+epsilon_k*(c+alpha*Q_min)
                        self.Q[i,u] ={'k':k+1, 'Q':Q_new}
                return 
                
        def findStarting(self, trjs, u , method='QL'):
                import numpy as np
                if u==1: # move up (pick up point as starting point)
                        index = np.argmax(trjs[1])
                elif u==2: # move down (pick down point as starting point)
                        index = np.argmin(trjs[1])
                elif u==3: # move right (pick right point as starting point)
                        index = np.argmin(trjs[0])
                elif u==4: # move lest (pick left point as starting point)
                        index = np.argmax(trjs[0])        
                newPoints = [[trjs[0][index]], [trjs[1][index]]]
                return newPoints

        def run(self, inits, nstepmax = 10):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                inits_x = inits[0]
                inits_y = inits[1]                    
                # temperature
                mu = 3
                # parameter used as stopping criterion
                tol1 = 1e-7
                n2 = 1e1
                n1 = len(inits_x)
                
                # time-step (limited by the ODE step)
                h = 1e-4

                # initialization                
                x = np.array(inits_x)
                y = np.array(inits_y)
                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)
                dx[0] = 0
                dy[0] = 0
                xi = x
                yi = y
                # parameters in Mueller potential
                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength

                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y
                zxx = np.mgrid[-1:2.51:0.01]
                zyy = np.mgrid[-1:2.51:0.01]
                xx, yy = np.meshgrid(zxx, zyy)
                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,5):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))
                ##### Main loop
                trj_x = [inits[0][0]]
                trj_y = [inits[1][0]]
                for nstep in range(int(nstepmax)):
                        # calculation of the x and y-components of the force, dVx and dVy respectively
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        dVx = (aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+cc[0]*(y-YY[0]))*ee
                        for j in range(1,5):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1)
                        trj_x.append(x[0][0]) 
                        trj_y.append(y[0][0])
                return trj_x, trj_y    

        def run_Plt(self, inits, nstepmax = 10):
                import numpy as np
                import time 
                from scipy.interpolate import interp1d
                import matplotlib.pyplot as plt
                inits_x = inits[0]
                inits_y = inits[1]                    
                plt.ion()
                # max number of iterations
                
                # frequency of plotting
                nstepplot = 1e1
                # plot string every nstepplot if flag1 = 1 
                flag1 = 1

                # temperature 
                mu =3
                # parameter used as stopping criterion
                tol1 = 1e-7
                n2 = 1e1;
                n1 = len(inits_x)
                # time-step 
                h = 1e-4
                
                # initialization
                x = np.array(inits_x)
                y = np.array(inits_y)
                dx = x-np.roll(x, 1)
                dy = y-np.roll(y, 1)
                dx[0] = 0
                dy[0] = 0
                xi = x
                yi = y
                # parameters in Mueller potential
                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength

                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y

                zxx = np.mgrid[-1:2.51:0.01]
                zyy = np.mgrid[-1:2.51:0.01]
                xx, yy = np.meshgrid(zxx, zyy)

                V1 = AA[0]*np.exp(aa[0] * np.square(xx-XX[0]) + bb[0] * (xx-XX[0]) * (yy-YY[0]) +cc[0]*np.square(yy-YY[0]))
                for j in range(1,5):
                        V1 =  V1 + AA[j]*np.exp(aa[j]*np.square(xx-XX[j]) + bb[j]*(xx-XX[j])*(yy-YY[j]) + cc[j]*np.square(yy-YY[j]))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.axvline(x=self.theta_mean[0])
                plt.axhline(y=self.theta_mean[1])
                plt.xlabel('x')
                plt.ylabel('y')
                ##### Main loop
                trj_x = []
                trj_y = []
                for nstep in range(int(nstepmax)):
                        # calculation of the x and y-components of the force, dVx and dVy respectively
                        ax.contourf(xx,yy,np.minimum(V1,200), 40)
                        ee = AA[0]*np.exp(aa[0]*np.square(x-XX[0])+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*np.square(y-YY[0]))
                        dVx = (aa[0]*(x-XX[0])+bb[0]*(y-YY[0]))*ee
                        dVy = (bb[0]*(x-XX[0])+cc[0]*(y-YY[0]))*ee
                        for j in range(1,5):
                                ee = AA[j]*np.exp(aa[j]*np.square(x-XX[j])+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*np.square(y-YY[j]))
                                dVx = dVx + (aa[j]*(x-XX[j])+bb[j]*(y-YY[j]))*ee
                                dVy = dVy + (bb[j]*(x-XX[j])+cc[j]*(y-YY[j]))*ee
                        x0 = x
                        y0 = y
                        x = x - h*dVx + np.sqrt(h*mu)*np.random.randn(1,n1)
                        y = y - h*dVy + np.sqrt(h*mu)*np.random.randn(1,n1) 
                        trj_x.append(x) 
                        trj_y.append(y)
                        for j in range(len(trj_x)):
                                ax.plot(trj_x[j], trj_y[j], 'o', color='w')
                        ax.plot(x,y, 'o', color='r')
                        fig.canvas.draw()
                return trj_x, trj_y          

        def pltFinalPoints(self, trjs_theta, num):
                import numpy as np
                import matplotlib.pyplot as plt
                plt.rcParams.update({'font.size':18})
                plt.rc('xtick', labelsize=18)
                plt.rc('ytick', labelsize=18)
                x = np.array(trjs_theta[0])
                y = np.array(trjs_theta[1])

                # parameters in Mueller potential
                aa = [-2, -20, -20, -20, -20] # inverse radius in x
                bb = [0, 0, 0, 0, 0] # radius in xy
                cc = [-20, -20, -2, -20, -20] # inverse radius in y
                AA = 30*[-200, -120, -200, -80, -80] # strength

                XX = [1, 0, 0, 0, 0.4] # center_x
                YY = [0, 0, 1, 0.4, 0] # center_y
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
                plt.plot(x, y, 'o', color='white', alpha=0.2, mec="black")
                plt.savefig('fig_'+str(num)+'.png', dpi =500)
                
