import numpy as np
import pyomo.environ as pyenv
import matplotlib.pyplot as plt
import pickle
import time as tm

import sys
# sys.path.insert(1,'C:/Users/Natha/bluesky/bluesky/plugins/MER_LMP_DATA')
import pandas as pd
# New York
# ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
dist = np.array([[0.,          10.48113508,  18.51467981,  18.54127528,  5.7385456,    13.03650962,  20.09902115],
  				[10.48113508,  0.,           9.15619145,   15.04929904,  10.56004164,  6.52505341,   13.22524436],
  				[18.51467981,  9.15619145,   0.,           11.47944457,  16.1038714,   6.31668958,   6.33708216 ],
  				[18.54127528,  15.04929904,  11.47944457,  0.,           13.32823709,  8.5872692,    6.16996146 ],
  				[5.7385456,    10.56004164,  16.1038714,   13.32823709,  0.,           9.86908414,   16.01052999],
  				[13.03650962,  6.52505341,   6.31668958,   8.5872692,    9.86908414,   0.,           7.31033037 ],
  				[20.09902115,  13.22524436,  6.33708216,   6.16996146,   16.01052999,  7.31033037,   0.         ]])

def global_optimization(
    T_horizon, num_agents, dist, LMP_NP=None, MER_NP=None, mip_gap=0.1, start_cust=10, save_flag=False, with_emis=True,
    dpm=0.5, g_i=250, E_max=100, E_discharge=4, capacity=4, mer_dpt=185, max_charge_duration=50,
    velocity=100, display_progress=False):

	'''
	Units Matter!
	LMP_NP -> numpy array for locational marginal price - $/mwh    - converted to $/kwh
	MER_NP -> numpy array for marginal emission rate    - tons/mwh - converted to tons/kwh - then mer_dpt converts it to $/kwh
	dpm    -> dollar per mile to calculated the fare charge
	g_i    -> charge rate - kw
	E_max  -> energy capacity of evtol - kwh
	E_discharge -> rate of discharge - kwh/mile
	velocity -> miles per hour - should maybe be knots by its okay for now

	'''

	LMP_df = pd.read_excel(r'/Users/duong/Work-Documents/NMSU/NASA_ULI_2022/my-code-no-public/evtol-scheduling/trip-noRQtime-formulation/v2-greedy-based/utils/MER_LMP_Information.xlsx', sheet_name = "LMP")
	MER_df = pd.read_excel(r'/Users/duong/Work-Documents/NMSU/NASA_ULI_2022/my-code-no-public/evtol-scheduling/trip-noRQtime-formulation/v2-greedy-based/utils/MER_LMP_Information.xlsx', sheet_name = "MER")
	LMP_NP = LMP_df.to_numpy().T
	MER_NP = MER_df.to_numpy().T
	LMP_NP = LMP_NP[1:]/1000
	MER_NP = MER_NP[1:]/1000


	num_cities = len(dist)

	# print("Max distance:",np.max(dist))
	# print("Max flight time in minutes:",np.max(dist)/velocity*60)
	# print("Min distance:",np.min(dist[np.nonzero(dist)]))
	# print("Min flight time in minutes:",np.min(dist[np.nonzero(dist)])/velocity*60)
	# print("Beginning solve...")


	start_loc = np.arange(num_agents)%num_cities + 1
	max_d = E_max/E_discharge
	V_max = capacity
	# mer_dpt = 185 #MER dollar per ton
	# max_charge_duration = 50


# determines the number of steps to solve for to encompas the whole time horizon
	min_trav_time = np.min(dist[np.nonzero(dist)])/velocity*60
	# mean_trav_time = np.mean(dist[np.nonzero(dist)])/velocity*60
	# max_trav_time = np.max(dist[np.nonzero(dist)])/velocity*60
	min_steps = int(np.floor(T_horizon/min_trav_time))
	# mean_steps = int(np.floor(T_horizon/mean_trav_time)) + 2
	# max_steps = int(np.floor(T_horizon/max_trav_time))
	N_steps = min_steps # 40 good value for 120 minutes...and it still solves -> 22 seems to be the max dang it
	# print("min steps:", min_steps)
	# print("mean steps:", mean_steps)
	# print("max steps:", max_steps)



	M = pyenv.ConcreteModel()
	M.i = pyenv.RangeSet(num_cities)
	M.N = pyenv.RangeSet(N_steps)
	M.k = pyenv.RangeSet(num_agents)

	M.w_mat = pyenv.Var(M.N, M.i, M.i, M.k, within = pyenv.Integers, bounds = (0,V_max)) # integer value of how many customers are served for each time step n
	M.charge_time_vec = pyenv.Var(M.N, M.i, M.k, bounds = (0,500)) # charge time in minutes for each time instance n

	M.x = pyenv.Var(M.N, M.i, M.i, M.k, within = pyenv.Binary) # 1 if edge is traveled 
	M.rem_energy = pyenv.Var(M.N, M.k, bounds = (0,E_max)) # remaining battery charge in kwh of the agent

	W = start_cust*(np.ones(num_cities)-np.eye(num_cities)) # constant matrix for reference purposes
	M.W_ij = pyenv.Var(M.N, M.i, M.i, within=pyenv.Integers, bounds=(0,start_cust))# The variable associated with how many customers are awaiting pickup at each vertiport W_ij[n,1,4]=2 means 2 people want to travel from vertiport 1 to 4 at time n

	M.wait_time = pyenv.Var(M.N, M.k, bounds = (0,T_horizon)) # 1 if edge is traveled 
	M.dep_time = pyenv.Var(M.N, M.k, bounds = (0,T_horizon)) # 1 if edge is traveled 


	# ==================================================================================================================================================================
	def obj_fcn(M):
		revenue_lin = sum(dpm*dist[i-1,j-1]*M.w_mat[n,i,j,k]for i in M.i for j in M.i for n in M.N for k in M.k)
		chg_cost_lin = sum(g_i*LMP_NP[n-1,i-1]/60*M.charge_time_vec[n,i,k] for i in M.i for n in M.N for k in M.k)
		if with_emis:
			emis_cost_lin = sum(mer_dpt*g_i/60*MER_NP[n-1,i-1]*M.charge_time_vec[n,i,k] for i in M.i for n in M.N for k in M.k)
		else:
			emis_cost_lin = 0
			
		return  revenue_lin - chg_cost_lin - emis_cost_lin
	M.objective = pyenv.Objective(rule = obj_fcn, sense = pyenv.maximize)


	# limits choice to one per agent
	def n_choice(M,n,k):
		return sum(M.x[n,i,j,k] for i in M.i for j in M.i)==1
	M.n_choice = pyenv.Constraint(M.N, M.k, rule=n_choice)

	# makes sure consecutive trips form a valid path between each time step
	def consec_steps(M,n,j,k):
		if n > 1:
			return sum(M.x[n-1,i,j,k] for i in M.i) == sum(M.x[n,j,i,k] for i in M.i)
		else:
			return sum(M.x[n,start_loc[k-1],j,k]for j in M.i)==1
	M.consec = pyenv.Constraint(M.N,M.i,M.k, rule = consec_steps)

	# charge of the agent for each time step
	def E_nk(M,n,k):
		if n >1:
			discharge = E_discharge*sum(dist[i-1,j-1]*M.x[n,i,j,k] for i in M.i for j in M.i)
			charge_lin = sum(g_i/60*M.charge_time_vec[n,i,k] for i in M.i)
			return M.rem_energy[n,k] == M.rem_energy[n-1,k] + charge_lin - discharge
		else:
			return M.rem_energy[1,k] == E_max
	M.E_nk = pyenv.Constraint(M.N, M.k, rule = E_nk)



	def charge_dur(M,n,i,k):
		return M.charge_time_vec[n,i,k] <= M.x[n,i,i,k]*max_charge_duration
	M.chrg_dur = pyenv.Constraint(M.N, M.i, M.k, rule=charge_dur)




	def pickup(M,n,i,j,k):
		if i !=j:
			return M.w_mat[n,i,j,k] <= V_max*M.x[n,i,j,k] 
		else:
			return M.w_mat[n,i,j,k] == 0
	M.pickup = pyenv.Constraint(M.N,M.i, M.i,M.k, rule =pickup)


	def w_consec(M,n,i,j):
		if n >1:
			return M.W_ij[n,i,j] == M.W_ij[n-1,i,j] - sum(M.w_mat[n,i,j,k] for k in M.k)
		else:
			return M.W_ij[n,i,j] == W[i-1,j-1] - sum(M.w_mat[n,i,j,k] for k in M.k)
	M.W_consec = pyenv.Constraint(M.N, M.i, M.i, rule = w_consec)


	# reassigned to be the time stamp on the steps
	def depart_time(M, n,k):
		if n>1:
			past_travel_time = sum(dist[i-1,j-1]*M.x[n-1,i,j,k]for i in M.i for j in M.i)/velocity*60
			return M.dep_time[n,k] == M.dep_time[n-1,k] + sum(M.charge_time_vec[n-1,i,k] for i in M.i) + past_travel_time + M.wait_time[n-1,k]
		else:
			return M.dep_time[n,k] == M.wait_time[n,k]
	M.depart_time = pyenv.Constraint(M.N, M.k, rule = depart_time)




	# ================================================================================================================================




	solver= pyenv.SolverFactory('gurobi')
	solver.options['MIPGap'] = mip_gap
	# solver.options['OutputFlag'] = 0
	start = tm.time()
	results =  solver.solve(M, warmstart = False, tee=display_progress)
	solve_time = tm.time() - start
	# print(results)

	if results.solver.status == "ok":
		X = np.zeros((N_steps, num_cities, num_cities, num_agents))
		for n in range(N_steps):
			for i in range(num_cities):
				for j in range(num_cities):
					for k in range(num_agents):
						X[n,i,j,k] = M.x[n+1,i+1,j+1,k+1].value

		trav_time = np.zeros((N_steps,num_agents))
		for n in range(N_steps):
			for k in range(num_agents):
				trav_time[n,k] = sum(dist[i,j]*M.x[n+1,i+1,j+1,k+1].value for i in range(num_cities) for j in range(num_cities))/velocity*60


		X_hist = np.zeros((num_agents, N_steps))
		for n in range(N_steps):
			for k in range(num_agents):
				col = max(X[n,:,:,k].argmax(axis=1)) +1
				row = np.argmax(X[0,:,:,0].argmax(axis=1))
				X_hist[k,n] = col
		
		# print("Flight dispatxh schedule for each agent at each time step:\n",X_hist)



		revenue_hist = np.zeros((num_agents, N_steps))
		for n in range(N_steps):
			for k in range(num_agents):
				revenue_hist[k,n] = round(100*sum(dpm*dist[i,j]*M.w_mat[n+1, i+1, j+1, k+1].value for i in range(num_cities) for j in range(num_cities) ))/100
		# print("Revenue for each each agent at each time instant:\n",revenue_hist)

		time_charge = np.zeros((num_agents, N_steps))
		cost_charge = np.zeros((num_agents, N_steps))
		cost_emis = np.zeros((num_agents, N_steps))
		cum_cost_charge = np.zeros(N_steps)
		cum_cost_emis = np.zeros(N_steps)
		c_charg = 0
		c_emis = 0
		lmp_hist = np.zeros((num_agents, N_steps))
		mer_hist = np.zeros((num_agents, N_steps))
		for k in range(num_agents):
			for n in range(N_steps):
				# time_charge[k,n] = sum(M.charge_time_mat[n+1, i+1, j+1, k+1].value for i in range(num_cities) for j in range(num_cities) )
				time_charge[k,n] = sum(M.charge_time_vec[n+1, i+1, k+1].value for i in range(num_cities) )
				# lmp = sum(LMP[n,i,j]*X[n,i,j,k] for i in range(num_cities) for j in range(num_cities))
				lmp = sum(LMP_NP[n,i]*X[n,i,i,k] for i in range(num_cities))
				# mer = sum(MER[n,i,j]*X[n,i,j,k] for i in range(num_cities) for j in range(num_cities))
				mer = sum(MER_NP[n,i]*X[n,i,i,k] for i in range(num_cities))
				cost_charge[k,n] = lmp*g_i/60*time_charge[k,n]
				cost_emis[k,n] = mer_dpt*mer*g_i/60*time_charge[k,n]
				# this is going to store the mer and lmp values for the instances that the agents are charging - then use this to find the cumulative mer and lmp
				if time_charge[k,n] != 0:
					lmp_hist[k,n] = lmp
					mer_hist[k,n] = mer


		# print("Cost charge:", cost_charge)
		
		# print("\n\n\n COST EMIS DEBUG:\n")
		# print(cost_emis)
		# print("\n\n\n")


		# print("\n\nTime scheduled to charge at each time instance:\n",time_charge)
		# print("\nTime of travel at each time instance:\n",trav_time)



		
		W_hist = np.zeros((N_steps, num_cities, num_cities))
		for n in range(N_steps):
			for i in range(num_cities):
				for j in range(num_cities):
					W_hist[n,i,j] = np.round(M.W_ij[n+1, i+1, j+1].value)
		# for n in range(N_steps):
			# print(W_hist[n,:,:])

		time_axis = np.zeros((N_steps, num_agents))
		for n in range(N_steps):
			for k in range(num_agents):
				time_axis[n,k] = round(M.dep_time[n+1, k+1].value)

		# print(revenue_hist)
		# print("\n\nTIME AXIS:\n",time_axis)

		# checks that each agent has been schudeuled up till the presribed time horizon
		sched_termination = np.min(time_axis[-1,:])
		# print("\nTermination time on schedule:", sched_termination)
		

	#####################################################
		charge_demand = np.zeros((N_steps, num_cities))
		for n in range(N_steps):
			for i in range(num_cities):
				for k in range(num_agents):
					if M.charge_time_vec[n+1, i+1, k+1].value > 0:
						charge_demand[n,i] += g_i/1000

		# print("Charge Demand:\n", charge_demand[:,0])


		

		valid_axis = np.arange(T_horizon+1) # integer values for the time axis - time was rounded to be 
		reven = np.zeros(T_horizon+1)
		charge = np.zeros(T_horizon+1)
		emis = np.zeros(T_horizon+1)
		cum_lmp_hist = np.zeros(T_horizon+1)
		cum_mer_hist = np.zeros(T_horizon+1)

		chg_dem_hist = np.zeros((T_horizon+1, num_cities))

		for entry in valid_axis:
			for k in range(num_agents):
				for n in range(N_steps):
					if time_axis[n,k] == entry:
						reven[entry] += revenue_hist[k,n]
						charge[entry] += cost_charge[k,n]
						emis[entry] += cost_emis[k,n]
						cum_lmp_hist[entry] += lmp_hist[k,n] # cumulative lmp/mer which gives insight into the total lmp/mer effects on the system
						cum_mer_hist[entry] += mer_hist[k,n]
						
						for i in range(num_cities):
							chg_dem_hist[entry, i] += charge_demand[n,i]

		# print("Charge demand history:\n", chg_dem_hist[:,0])

		#######################################################################


		# print("\n\ntotal revenue added at each minute:\n", reven)
		cum_rev = np.zeros(T_horizon+1)
		cum_charge = np.zeros(T_horizon+1)
		cum_emis = np.zeros(T_horizon+1)
		eff_lmp = np.zeros(T_horizon+1)
		eff_mer = np.zeros(T_horizon+1)
		rev_tracker = 0
		ch_cost_tracker = 0
		em_cost_tracker = 0
		lmp_tracker = 0
		mer_tracker = 0
		for entry in valid_axis:
			rev_tracker += reven[entry]
			ch_cost_tracker += charge[entry]
			em_cost_tracker += emis[entry]
			lmp_tracker += cum_lmp_hist[entry]
			mer_tracker += cum_mer_hist[entry]
			cum_rev[entry] = rev_tracker
			cum_charge[entry] = ch_cost_tracker
			cum_emis[entry] = em_cost_tracker
			eff_lmp[entry] = lmp_tracker
			eff_mer[entry] = mer_tracker


	
		# if sched_termination < T_horizon:
		# 	print("\n\n\n\n!!!!!!!!!!!!!!!! Scheduling incomplete, add more steps to horizon. !!!!!!!!!!!!!!!!!!!\n\n\n")




		# print("Solved in (ONLY) %0.2f seconds!"%(solve_time))
		# print("Solved in (ONLY) %0.2f minutes!"%(solve_time/60))

		
		dest_hist = np.zeros((N_steps,num_agents))
		chg_times = np.zeros((N_steps,num_agents))
		dep_times = np.zeros((N_steps,num_agents))
		energy_hist = np.zeros((N_steps,num_agents))
		cust_hist = np.zeros((N_steps,num_agents))


		for n in range(N_steps):
			for k in range(num_agents):
				dest_hist[n,k]   = np.argmax(X[n,:,:,k].sum(axis=0))
				chg_times[n,k]   = time_charge[k,n]
				dep_times[n,k]   = M.dep_time[n+1,k+1].value
				energy_hist[n,k] = M.rem_energy[n+1,k+1].value
				cust_hist[n,k]   = sum(M.w_mat[n+1, i+1, j+1, k+1].value for i in range(num_cities) for j in range(num_cities))


		total_prof = cum_rev[-1] - cum_charge[-1] - cum_emis[-1]
		if save_flag == True:
			print("Soaking Cucumbers")
			file_name = "global_optimization_"+str(num_agents)+"agents_"+str(mip_gap)+"mip_gap"+str(start_cust)+"cust"
			if with_emis == True:
				with open(file_name+"_emis.txt",'wb') as f1:
					# pickle.dump([valid_axis, cum_rev, cum_charge, cum_emis, eff_lmp, eff_mer], f1)
					pickle.dump([solve_time, total_prof], f1)
			if with_emis == False:
				with open(file_name+"_no_emis.txt",'wb') as f1:
					# pickle.dump([valid_axis, cum_rev, cum_charge, cum_emis, eff_lmp, eff_mer], f1)
					pickle.dump([solve_time, total_prof], f1)
			print("Pickles are ready for eating!")


		return solve_time, total_prof
	


		
t, prof = global_optimization(T_horizon=30, dist=dist, num_agents = 10, mip_gap = 0.1, display_progress=True, save_flag=False)