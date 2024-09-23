from global_async_function import global_optimization as GO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


dist = np.array([[0.,          10.48113508,  18.51467981,  18.54127528,  5.7385456,    13.03650962,  20.09902115],
  				[10.48113508,  0.,           9.15619145,   15.04929904,  10.56004164,  6.52505341,   13.22524436],
  				[18.51467981,  9.15619145,   0.,           11.47944457,  16.1038714,   6.31668958,   6.33708216 ],
  				[18.54127528,  15.04929904,  11.47944457,  0.,           13.32823709,  8.5872692,    6.16996146 ],
  				[5.7385456,    10.56004164,  16.1038714,   13.32823709,  0.,           9.86908414,   16.01052999],
  				[13.03650962,  6.52505341,   6.31668958,   8.5872692,    9.86908414,   0.,           7.31033037 ],
  				[20.09902115,  13.22524436,  6.33708216,   6.16996146,   16.01052999,  7.31033037,   0.         ]])
# separate experiments:

# 1
# see how solve time is is affected by increasing number of evtol
# see how profit is affected by increasing evtol

# 2
# see how solve time is affects by decreasing mip gap
# see how profit is affects by decreasing mip gap

# two values that change - mip_gap, num_evtol
# two values I car about - profit , solve time

min_agents = 5
interval_agent = 2 
max_agents = 35

# TOTD: run from .20 gap on
# max_gap = 12 # percent
max_gap = 4 # percent
interval_gap = 1
min_gap = 2  # percent

num_agents_list = np.arange(min_agents, max_agents+interval_agent, interval_agent)
mip_gap_list = (np.arange(min_gap, max_gap+interval_gap, interval_gap))/100
mip_gap_list = mip_gap_list[::-1] # reverse the mip gap list so that we solve the faster (larger) mip gap solutions first
print(mip_gap_list)
print(num_agents_list)

time_data = np.zeros((len(num_agents_list),len(mip_gap_list)))
prof_data = np.zeros((len(num_agents_list),len(mip_gap_list)))


col = 0
for mip_gap in mip_gap_list:
	row = 0
	for num_agents in num_agents_list:
		print("\n\n\n#######################################################")
		print("%d agents   |    %0.2f mip gap"%(num_agents, mip_gap))
		print("#######################################################")
		t, prof = GO(T_horizon=180, dist=dist, num_agents = num_agents, mip_gap = mip_gap, display_progress=True, save_flag=False)
		print("time:", t)
		print("profit:", prof)
		time_data[row,col] = t
		prof_data[row,col] = prof
		
		row+=1
	col+=1


fig= plt.figure()
ax = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122,projection='3d')
y = num_agents_list
x = mip_gap_list[::-1]
X,Y = np.meshgrid(x,y)
Z1 = time_data
Z2 = prof_data
# print("x:",x)
# print("y:",y)
# print("z:",z1)

surf = ax.plot_surface(X,Y,Z1, linewidth=1, antialiased=False)
surf2 = ax2.plot_surface(X,Y,Z2, linewidth=1, antialiased=False)
ax.set_ylabel("Number eVTOL")
ax.set_xlabel("MIP Gap")
ax.set_zlabel("Solve Time (seconds)")
ax2.set_ylabel("Number eVTOL")
ax2.set_xlabel("MIP Gap")
ax2.set_zlabel("Profit ($)")
plt.show()
