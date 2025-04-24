import sys
import random
import os
import numpy as np
vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
# vPort_id = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]

def InitLoc(nAgents, seed, limit_nAgents_each_vertiport):
    atom = ''
    part=[]
    def get(val, default):
        return val if val != None else default

    # input arguments, need nAgents and seed

    random.seed(seed)
    fName = 'init.lp'
    decoupled_init_fName = 'init_decoupled.lp'
    #init location
    #discharge rate
    #emax
    #emin

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
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())
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



    
def GenRq(cust, seed):

    random_flag = 0
    out_filename = 'rq.lp'



    request_p = [[j, i] for i in range(0, 7) for j in range(0, 7) if i != j]
    matrix_out = [[random.randint(0, cust) if i!=j else 0 for j in range(0, len(vPort_id))] for i in range(0, len(vPort_id))]

    request_p = [[vPort_id[m[0]], vPort_id[m[1]]]for m in request_p]
    request_print = []
    total_edge = len(vPort_id) * (len(vPort_id)-1)
    for i in range(0, total_edge):
        no = i
        origin = request_p[i][0] 
        destination = request_p[i][1]
        if random_flag == 0:
            request_print += [[no, (origin, destination), cust]]
        else:
            cust_rand = matrix_out[vPort_id.index(origin)][vPort_id.index(destination)]
            request_print += [[no, (origin, destination), cust_rand]]
    # Print the current working directory
    print("Current Working Directory:", os.getcwd()) 
    f = open(f'instances/rq.lp',"w+")
    f.write(f'%{matrix_out}')
    f.write('%request(ID, (edge), number of request passenger)\n')
    for i in request_print:
        f.write('request' + str(tuple(i)).replace("'","") + '.\n')

    f.close()