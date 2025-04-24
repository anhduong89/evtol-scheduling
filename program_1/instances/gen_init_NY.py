import sys
import random
import os
import numpy as np

atom = ''
part=[]
def get(val, default):
    return val if val != None else default



nAgents = int(sys.argv[1])
fName = 'instances/init.lp'
#init location
#discharge rate
#emax
#emin
# vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
vPort_id = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]
emax_print = [[i, 100] for i in range(0, nAgents)]
emin_print = [[i, 0] for i in range(0, nAgents)]
dischg_rate_print = [[i, 4] for i in range(0, nAgents)]
capacity_print = [[i, 4] for i in range(0, nAgents)]
init_loc_print = []
init_loc = np.arange(nAgents)%(len(vPort_id))
for i in range(0, nAgents):
    init_loc_print += [[i, vPort_id[init_loc[i]]]]


init_battery_print = []
for i in range(0, nAgents):
    init_battery_print += [[i, random.choice(range(0, 50))]]
    
    
variable = [init_loc_print, emax_print, emin_print, dischg_rate_print, capacity_print, init_battery_print]
atoms_name = ['init_loc', 'emax', 'emin', 'dischg_rate', 'capacity', 'init_battery']


f = open(fName,"w+")
#AGENT
f.write('agent('+str(0)+'..'+str(nAgents-1)+').\n')
for var, atom_name in zip(variable, atoms_name):
    for i in var:
        f.write(atom_name + str(tuple(i)).replace("'","") + '.\n')
# # init
# for i in init_loc_print:
#     f.write('init_loc' + str(tuple(i)).replace("'","") + '.\n')
# #emax

# for i in emax_print:
#     f.write('emax' + str(tuple(i)).replace("'","") + '.\n')
    
# #emin

# for i in emin_print:
#     f.write('emin' + str(tuple(i)).replace("'","") + '.\n')
    
# #dischg_rate

# for i in dischg_rate_print:
#     f.write('dischg_rate' + str(tuple(i)).replace("'","") + '.\n')

# for i in capacity_print:
#     f.write('capacity' + str(tuple(i)).replace("'","") + '.\n') 

f.close()


    
    