import sys
import random
import os
import numpy as np

atom = ''
part=[]
def get(val, default):
    return val if val != None else default


# input arguments, need nAgents and seed
nAgents = int(sys.argv[1])
seed = int(sys.argv[2])
limit_nAgents_each_vertiport = int(sys.argv[3])
random.seed(seed)
fName = 'instances/init.lp'
decoupled_init_fName = 'instances/init_decoupled.lp'
#init location
#discharge rate
#emax
#emin
# vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
vPort_id = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]
emax_print = [[i, 60] for i in range(0, nAgents)]
emin_print = [[i, 0] for i in range(0, nAgents)]
dischg_rate_print = [[i, 4] for i in range(0, nAgents)]
capacity_print = [[i, 4] for i in range(0, nAgents)]
init_loc_print = []
# init_loc = np.arange(nAgents)%(len(vPort_id))
init_loc = []
counts = {}
for i in range(nAgents):
    while True:
        val = random.randint(0, len(vPort_id)-1)
        if counts.get(val, 0) < limit_nAgents_each_vertiport:
            counts[val] = counts.get(val, 0) + 1
            init_loc.append(val)
            break
for i in range(0, nAgents):
    init_loc_print += [[i, vPort_id[init_loc[i]]]]


counts = {}  # Dictionary to count occurrences of each random number


init_battery_print = []
for i in range(0, nAgents):
    init_battery_print += [[i, 60]]
    
    
variable = [init_loc_print, emax_print, emin_print, dischg_rate_print, capacity_print, init_battery_print]
atoms_name = ['init_loc', 'emax', 'emin', 'dischg_rate', 'capacity', 'b_init']

# new_file_name = "instances/recorded_init/init_{seed}.lp".format(seed = seed)
f = open(decoupled_init_fName,"w+")
#AGENT
# f.write('agent('+str(0)+'..'+str(nAgents-1)+').\n')
f.write(f"%battery in minute.\n")
for var, atom_name in zip(variable, atoms_name):
    for i in var:
        f.write(atom_name + str(tuple(i)).replace("'","") + '.\n')
f.close()



backup_f = open(fName,"w+")
#AGENT
backup_f.write('agent('+str(0)+'..'+str(nAgents-1)+').\n')
for var, atom_name in zip(variable, atoms_name):
    for i in var:
        backup_f.write(atom_name + str(tuple(i)).replace("'","") + '.\n')
backup_f.close()
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



    
    