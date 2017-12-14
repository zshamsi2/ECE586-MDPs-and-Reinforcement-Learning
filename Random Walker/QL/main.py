import walkerSimEnv as sim
import numpy as np 
import pickle

ITERMAX = 100
l = 100

my_sim = sim.walkerSimulation()
totalTimes = []
totalCosts = []
for epoc in range(5):
	print('epoc '+str(epoc)+' is running...')
	totalCost = 0
	# run first round of simulation
	X_0 = [1.52]
	Y_0 = [0.36]
	trjs = my_sim.run([X_0, Y_0], nstepmax = l) # 2xl
	i = my_sim.updateState(trjs)
	k = 1
	for round in range(ITERMAX):
		# find a value for control 
		u = my_sim.findU(i, k)
		newPoints = my_sim.findStarting(trjs, u)
	
		trj1 = my_sim.run(newPoints, nstepmax = l) # trj with shape of [[Xs][Ys]]
		com_trjs = []
		for theta in range(len(trj1)):
			com_trjs.append(np.concatenate((trjs[theta], trj1[theta]))) #check dim
		trjs = np.array(com_trjs)
		# update the state of the system
		j = my_sim.updateState(trjs) # based on all trajectories
		
		# calculate imidiate cost
		cost = my_sim.cost(i, j)
		totalCost = totalCost+cost
		# update Q function
		my_sim.updateQ(i, j, u, cost) # important
		i = j
		
		#if my_sim.isFinal(com_trjs[1]):
		if  np.any(np.array(com_trjs[1])>1.5): # terminate condition
			totalTime = l*round
			print('The simulation reached the final point in '+str(totalTime)+' steps')
			break
	totalTimes.append(totalTime)
	totalCosts.append(totalCost)
	#my_sim.pltFinalPoints(trjs, epoc)

	
pickle.dump(my_sim.Q, open('Q','wb'))
print('Q function learned: ', my_sim.Q)
np.savetxt('totalTimes', totalTimes)
np.savetxt('totalCosts', totalCosts)
