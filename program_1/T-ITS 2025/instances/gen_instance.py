import sys
import random
import os
import numpy as np
vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
# vPort_id = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]

def InitLoc(nAgents, seed=None, limit_nAgents_each_vertiport=6, out_file='instances/init.lp'):
    atom = ''
    part=[]
    def get(val, default):
        return val if val != None else default

    # input arguments, need nAgents and seed

    random.seed(seed)
    fName = 'instances/init.lp'
    decoupled_init_fName = 'instances/init_decoupled.lp'
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
    if seed != None:
        for i in range(nAgents):
            while True:
                val = random.randint(0, len(vPort_id)-1)
                if counts.get(val, 0) < limit_nAgents_each_vertiport:
                    counts[val] = counts.get(val, 0) + 1
                    init_loc.append(val)
                    break
        for i in range(0, nAgents):
            init_loc_print += [[i, vPort_id[init_loc[i]]]]
    else:
        for i in range(nAgents):
            init_loc_print += [[i, vPort_id[i%len(vPort_id)]]]


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



    backup_f = open(out_file,"w+")
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



    
def GenRq(cust=None, random_flag=False, out_filepath='instances/rq.lp'):


    request_p = [[j, i] for i in range(0, 7) for j in range(0, 7)]
    matrix_out = [[random.randint(0, cust) if i!=j else 0 for j in range(0, len(vPort_id))] for i in range(0, len(vPort_id))]

    request_p = [[vPort_id[m[0]], vPort_id[m[1]]]for m in request_p]
    request_print = []
    total_edge = len(vPort_id) * (len(vPort_id))
    for i in range(0, total_edge):
        no = i
        origin = request_p[i][0] 
        destination = request_p[i][1]
        if not random_flag:
            request_print += [[no, (origin, destination), cust]] if origin != destination else [[no, (origin, destination), 0]]
        else:
            cust_rand = matrix_out[vPort_id.index(origin)][vPort_id.index(destination)]
            request_print += [[no, (origin, destination), cust_rand]]

    f = open(out_filepath,"w+")
    f.write(f'%{matrix_out}')
    f.write('%request(ID, (edge), number of request passenger)\n')
    for i in request_print:
        f.write('request' + str(tuple(i)).replace("'","") + '.\n')
    f.close()
    
    if random_flag:
        return [req[2] for req in request_print]

def GenRqTime(total_cust=30, period=3, horizon=180):

    def generate_random_requests(total, peaks):
        res = [0] * (horizon//period)
        if total < peaks:
            raise ValueError("Total requests must be greater than or equal to the number of peaks.")
        peak_total = round(total * random.uniform(0.75, 0.8))
        peak_indices = random.sample(range(len(res) - 1), peaks)
        for i, peak in enumerate(peak_indices):
            peak_value = peak_total // peaks if i < peaks - 1 else peak_total - sum(res)
            res[peak] = peak_value // 2
            res[peak + 1] = peak_value - res[peak]
        remain = total - sum(res)
        while remain > 0:
            idx = random.choice([i for i, v in enumerate(res) if v == 0])
            add = random.randint(1, min(remain, max(res) // 2))
            res[idx] = add
            remain -= add
        return res

    def random_low(): return generate_random_requests(random.randint(25, 40), 2)
    def random_med(): return generate_random_requests(random.randint(40, 80), 3)
    def random_high(): return generate_random_requests(random.randint(80, 100), 4)


    with open("instances/rq_t.lp", "w") as f:
        total, req_id = 0, 0
        for v1 in vPort_id:
            for v2 in vPort_id:
                if v1 == v2: continue
                requests = random.choice([random_low, random_med, random_high])()
                scale_factor = total_cust / sum(requests)
                requests = [round(count * scale_factor) for count in requests]
                for t, count in enumerate(requests):
                    if count:
                        f.write(f"request({req_id},({v1},{v2}),{count},{t}).\n")
                        req_id += 1
                total += sum(requests)
    print("Total:", total, "\nDONE!!!")



if __name__ == '__main__':
    InitLoc(34)
    GenRq(30)
    # GenRqTime()