import walkerSimEnv as sim
import numpy as np 

X_0 = [1.52]
Y_0 = [0.36]

N = len(X_0) # number of parallel runs 
l = 10
# run first round of simulation
my_sim = sim.walkerSimulation()

# first round
trj1 = my_sim.run([X_0, Y_0], nstepmax = l)

trjs = trj1
trj_theta = my_sim.map(trjs)

newPoints = my_sim.findStarting(trj1_Sp_theta, starting_n = N , method = 'QL')

trjs_theta = trj1_Sp_theta

trjs_Sp_theta = trj1_Sp_theta
for round in range(700):
	# updates the std and mean 
	my_sim.updateState(trjs_theta) # based on all trajectories
	W_1 = my_sim.updateQ(trjs_theta) # important
	
	trj1 = my_sim.run_noPlt(newPoints, nstepmax = l)
	trj1 = my_sim.PreAll(trj1) # 2 x all points of this round

	com_trjs = []
	for theta in range(len(trj1)):
		com_trjs.append(np.concatenate((trjs[theta], trj1[theta])))
	
	trjs = np.array(com_trjs)
	trjs_theta = np.array(my_sim.map(trjs))
	trjs_Sp = my_sim.PreSamp(trjs, starting_n = N)
	trjs_Sp_theta = np.array(my_sim.map(trjs_Sp))

	newPoints = my_sim.findStarting(trjs_Sp_theta, trjs_Sp, W_1, starting_n = N , method = 'RL')
	
    
np.save('trjs_theta', trjs_theta)
