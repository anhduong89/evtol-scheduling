from solverClingoAPI import MSMAP, Schedule, ClingoComputeServedCustomers
from instances.gen_instance import GenRq, InitLoc
from utils.utils import EstimateMinAgents, CalProfit
import csv
import pandas as pd
import os
import openpyxl

# experiment of paper T-ITS
def RunExp0():
    results_file = 'results_exp0.csv'
    
    # # Erase the CSV file at the beginning if it exists
    # if os.path.exists(results_file):
    #     os.remove(results_file)

    columns = ["nb_agent", "nb_rq", "nb_segment", "exp_name", "served_customers", "profit", "run_time"]

    # # Write header once at the beginning
    # with open(results_file, mode='w', newline='') as file:
    #     writer = csv.DictWriter(file, fieldnames=columns)
    #     writer.writeheader()
    i_n = 0
    for nb_agent, nb_rq in [(10,10), (50,50), (50,100), (100,50), (100,100), (100,150)]:
        
        InitLoc(nAgents=nb_agent, out_file='T-ITS 2025/instances/init.lp')
        GenRq(cust=nb_rq, out_file='T-ITS 2025/instances/rq.lp')
        for alg in [("R", None, None), ("O1",[1],None), ("O2", [2], None), ("H1", None, [1]), ("H2", None, [2]), ("H1O1", [1], [1]), ("H2O2", [2], [2])]: 
            heuristic = alg[2] is not None
            for nb_segment in [7,8,9,10,11]:
                if i_n < 110:
                    i_n += 1
                    continue
                print(f" Running with experiment {i_n} algorithm {alg[0]} nb_agent={nb_agent}, nb_rq={nb_rq}, nb_segment={nb_segment}")
                s = Schedule(
                    time_limit=60,
                    update_time_limit=5,
                    max_segment=nb_segment,
                    encoding='T-ITS 2025/schedule.lp',
                    network='T-ITS 2025/instances/network_NY_0.lp',
                    rq_file='T-ITS 2025/instances/rq.lp',
                    init_file='T-ITS 2025/instances/init.lp',
                    choose_heu=alg[2],
                    choose_opt=alg[1],
                    dl_theory=True,
                    horizon=180,
                    heuristic=heuristic,
                )
                s.Solving()
                if s.models:
                    model = s.models[-1]
                else:
                    print(f"Warning: No answer set found. Skipping this configuration.")
                    i_n += 1
                    continue
                unserved_customers = ClingoComputeServedCustomers(model.flight_path_fact_format)
                result = {
                    "nb_agent": nb_agent,
                    "nb_rq": nb_rq,
                    "nb_segment": nb_segment,
                    "exp_name": alg[0],
                    "served_customers": nb_rq*42 - unserved_customers,
                    "profit": model.profit,
                    "run_time": model.run_time
                }
                print(result)
                # Append result to CSV file
                with open(results_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writerow(result)
                i_n += 1

def RunExp2():
    results = []
    for i in range(100):
        seed_init = i
        vert_cap = 10
        n_rq = 30
        cust_per_edge=GenRq(cust=30, random_flag=True, out_filepath=f'instances/100rand/rq_{i}.lp')
        
        min_agents, min_segments = EstimateMinAgents(horizon=180, b_init=15, demand_cust=cust_per_edge, aircraft_capacity=4)
        print('min_agents:', min_agents)
        print('min_segments:', min_segments)
        InitLoc(nAgents=min_agents, seed=seed_init, limit_nAgents_each_vertiport=vert_cap, out_file=f'instances/100rand/init_{seed_init}.lp')
        out = MSMAP(time_limit=30, max_segment=min_segments, network='instances/network_NY_0.lp', rq_file=f'instances/100rand/rq_{i}.lp',
            init_file=f'instances/100rand/init_{seed_init}.lp')
        # Initialize a list to store the results

        print(f'msmap_output: {out}')
        # Collect data for each iteration
        nonzero_cust_per_edge = [x for x in cust_per_edge if x != 0]
        result_row = {
            "i": i,
            "total_cust_per_edge": sum(cust_per_edge),
            "min_cust_per_edge": min(nonzero_cust_per_edge) if nonzero_cust_per_edge else 0,
            "max_cust_per_edge": max(cust_per_edge),
            "median_cust_per_edge": sorted(cust_per_edge)[len(cust_per_edge) // 2],
            "mean_cust_per_edge": sum(cust_per_edge) / len(cust_per_edge) if cust_per_edge else 0,
            "agents": min_agents,
            "segments": min_segments,
            "msmap_output": out  # Assuming MSMAP has an 'output' attribute or method
        }
        # Append to CSV after each iteration
        write_header = i == 0
        with open('results_exp2.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["i", "total_cust_per_edge", "min_cust_per_edge", "max_cust_per_edge", "median_cust_per_edge", "mean_cust_per_edge", "agents", "segments", "msmap_output"])
            if write_header:
                writer.writeheader()
            writer.writerow(result_row)

def RunExp1():
    for i in range(100):
        seed_init = i
        vert_cap = 10
        n_rq = 30
        
        InitLoc(nAgents=34, seed=seed_init, limit_nAgents_each_vertiport=vert_cap, out_file=f'instances/100rand_1/init_{seed_init}.lp')
        out = MSMAP(time_limit=30, max_segment=13, network='instances/network_NY_0.lp',
            init_file=f'instances/100rand_1/init_{seed_init}.lp')
        # Save i and MSMAP output to CSV
        with open('results_exp1.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, out])  # Assuming MSMAP has an 'output' attribute or method
        
if __name__ == "__main__":
    RunExp0()