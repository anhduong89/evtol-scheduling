from clingo import Control, Model
from clingo.ast import parse_string, ProgramBuilder
from clingodl import ClingoDLTheory
import sys
import argparse
import os
import time
import subprocess
from utils.utils import CalProfit, EstimateMinAgents
from instances.gen_instance import GenRq, InitLoc
import random
from typing import List
import re
import multiprocessing
import threading
import ctypes
# Callback function to handle models during solving
def on_model(m):
    print (m)

# Separator line for output readability
LINE = "---------------------------------------------------------------------"

# Uncommented argument parsing section for potential CLI usage
parser = argparse.ArgumentParser(description='Solver for eVTOL scheduling.')
parser.add_argument('--n_rq', type=int, default = 30, help='Number of requests')
parser.add_argument('--n_agents', type=int, default = 34, help='Number of agents')
parser.add_argument('--max_segment', type=int, default = 13, help='Maximum segment')
parser.add_argument('--horizon', type=int, default = 180, help='Horizon')
parser.add_argument('--time_limit', type=int, default = 30, help='Time limit')
parser.add_argument('--seed', type=int, default = 1, help='seed for random initial location')
parser.add_argument('--vertiport_cap', type=int, default = 6, help='vertiport capacity')
args = parser.parse_args()

# Estimate the minimum number of agents and segments required
# min_agents, min_segments = EstimateMinAgents(horizon=180, b_init=60, demand_cust=30, aircraft_capacity=4)

# Class to handle scheduling logic
class Schedule:
    def __init__(self, time_limit: int = 30, update_time_limit: int= None, start_segment: int = 1, max_segment: int = 13, horizon=None , 
                encoding: str = 'schedule.lp', network: str = None, init_file='instances/init.lp', rq_file ='instances/rq.lp', heuristic: bool = False, 
                choose_heu: list = None, choose_opt: list = None, 
                dl_theory: bool = False):
        self.time_limit = time_limit
        self.update_time_limit = update_time_limit
        self.start_segment = start_segment
        self.max_segment = max_segment
        self.horizon = horizon
        self.encoding = encoding
        self.network = network
        self.init = init_file
        self.rq = rq_file
        self.clingo_args = ['-c', f'start_seg={self.start_segment}']
        if horizon is not None:
            self.clingo_args.extend(['-c', f'horizon={self.horizon}'])
        if max_segment is not None:
            self.clingo_args.extend(['-c', f'max_seg={self.max_segment}'])
        if heuristic:
            self.clingo_args.append('--heuristic=Domain')
        self.choose_heu = choose_heu
        self.choose_opt = choose_opt
        self.dl_theory = dl_theory

    def Solving(self, fact_load=""):
        # Combine all input files into temporary file
        combined_program = ""
        
        # Read all input files
        with open(self.encoding, 'r') as f:
            combined_program += f.read() + "\n"
            
        if self.network:
            with open(self.network, 'r') as f:
                combined_program += f.read() + "\n"
                
        with open(self.init, 'r') as f:
            combined_program += f.read() + "\n"
            
        with open(self.rq, 'r') as f:
            combined_program += f.read() + "\n"

        # Add optimization and heuristic files
        if self.choose_opt:
            for opt in self.choose_opt:
                with open(f'T-ITS 2025/opt_heu/opt_{opt}.lp', 'r') as f:
                    combined_program += f.read() + "\n"

        if self.choose_heu:
            for heu in self.choose_heu:
                with open(f'T-ITS 2025/opt_heu/heu_{heu}.lp', 'r') as f:
                    combined_program += f.read() + "\n"

        # Add facts
        combined_program += fact_load + "\n"

        # Write combined program to temp file
        with open('temp_program.lp', 'w') as f:
            f.write(combined_program)

        # Construct clingo command
        cmd = ['clingo']
        
        if self.dl_theory:
            cmd[0] = 'clingo-dl'
        cmd.extend(['temp_program.lp', '--time-limit=' + str(self.time_limit), '-q1', '--outf=1', '-V1'])
        cmd.extend(self.clingo_args)
        try:
            # Run clingo
            print('Running clingo...')
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()

            # Clean up
            os.remove('temp_program.lp')

            if process.returncode != 0:
                print(f"Error running clingo: {stderr}")
                return []

            # Parse output and return models
            return self._parse_clingo_output(stdout)

        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    def _parse_clingo_output(self, output):
        # Parse clingo output and return list of models
        # This is a simplified parser - you may need to enhance it based on your needs
        models = []
        for line in output.split('\n'):
            if line.startswith('Answer'):
                model = argparse.Namespace()
                model.output = line
                models.append(model)
        print(models)
        return models
    
# Utility function to solve a Clingo program with additional facts
def ClingoSolver(prg_path, facts):
    with open(prg_path, "r") as file:
        prg = file.read()
    prg += facts
    ctl = Control()
    ctl.add("base", [], prg)
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            return " ".join([f"{str(symbol)}." for symbol in model.symbols(shown=True)])

def ClingoDLSolver(prg_path, facts):
    ctl = Control()
    thy = ClingoDLTheory()
    thy.register(ctl)
    with open(prg_path, "r") as file:
        prg = file.read()
    prg += facts
    with ProgramBuilder(ctl) as bld:
        parse_string(prg, lambda ast: thy.rewrite_ast(ast, bld.add))
    ctl.ground([("base", [])])
    thy.prepare(ctl)
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            print(model)

def ClingoComputeServedCustomers(flight_path):
    with open("T-ITS 2025/compute_revenue.lp", "r") as file:
        prg = file.read()
    prg += flight_path
    ctl = Control()
    ctl.add("base", [], prg)
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            for atom in model.symbols(shown=True):
                if atom.name == "total_remain":
                    return atom.arguments[0].number

# Get the model with the best optimization cost
def GetBestModelOptCost(models: List[argparse.Namespace]):
    if not models:
        return None
    return max(models, key=lambda model: model.opt_cost)

# Get the model with the highest profit
def GetBestModelProfit(models: List[argparse.Namespace]):
    if not models:
        return None
    return max(models, key=lambda model: model.profit)

# Compute revenue based on flight paths
def ComputeRevenue(flight_path):
    flight_path = " ".join([i + '.' for i in flight_path.split()])
    ctl = Control([])
    ctl.add(flight_path)
    ctl.load('compute_revenue.lp')
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as out:
        print(out.model())

# Get the number of agents at each vertiport
def GetNumberOfAgentsEachVertiport(dl_assignments):
    schedule = " ".join([i + '.' for i in dl_assignments])
    ctl = Control([])
    ctl.add(schedule)
    ctl.load('transfer.lp')
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as out:
        print(out.model())

# Centralized scheduling approach
def CentralSchedule():
    schedule = Schedule()
    models = schedule.Solving()
    best_model = GetBestModelProfit(models)
    return best_model

# Consecutive scheduling approach
def ConsecutiveSchedule():
    schedule = Schedule(choose_opt=[2, 2.1])
    models = schedule.Solving()
    best_model = GetBestModelOptCost(models)
    flight_path = best_model.flight_path_clingo_format
    
    schedule = Schedule(start_segment=11, max_segment=13, heuristic=False, choose_opt=[1, 2.1])
    print(flight_path)
    models = schedule.Solving(fact_load=flight_path)
    best_model = GetBestModelOptCost(models)

    return best_model

def DP():
    flight_path =''
    for d in range(0, 56):
        print(f'========== Schedule Agent {d}:')
        schedule = Schedule(time_limit=10
                            , start_segment=1
                            , max_segment=6
                            , encoding='encoding/d_s.lp'
                            , network='instances/network_NY.lp'
                            , init_path= 'instances/init_decoupled.lp'
                            , heuristic=True
                            , choose_heu=[2]
                            , choose_opt=[1]
                            , n_rq=None
                            , n_agents=None
                            , seed_init=1
                            , seed_rq=1
                            , dl_theory=True
                            )
        models = schedule.Solving(fact_load=f'agent({d}).'+ flight_path)
        best_model = GetBestModelOptCost(models=models)
        flight_path = best_model.flight_path_fact_format
        print(flight_path)
    return best_model.to_visualized

# Switching trajectory approach 1
def MSMAP(time_limit=30, start_segment=1, max_segment=13, network='instances/network_NY_0.lp',
                        rq_file='instances/rq.lp',
                        init_file='instances/init.lp'):
    # Initial solution
    print('=========== step 1: assign trajectory allowing operation time > horizon, prioritizing serving all customers:')
    s = Schedule(time_limit=time_limit
                , start_segment=start_segment
                , max_segment=max_segment
                , encoding='encoding/s.lp'
                , network=network
                , heuristic=True
                , choose_heu=None
                , choose_opt=None
                , init_path=init_file
                , rq_path=rq_file
                , dl_theory=False
                )
    models = s.Solving()
    best_model = min(models, key=lambda model: model.opt_cost)
    # best_model = GetBestModelOptCost(models=models)

    # Switching logic
    answer_set = best_model.flight_path_fact_format
    print(answer_set)
    print("========== step 2: start switching trajectory to constraint operation time <= horizon")
    sw = Schedule(time_limit=15
                , encoding='encoding/sw0.1.lp'
                , network=network
                , horizon=180
                , heuristic=False
                , choose_heu=None
                , choose_opt=None
                , dl_theory=False
                )
    models = sw.Solving(fact_load=answer_set)
    best_model = max(models, key=lambda model: model.opt_cost)


    model_after_switch = best_model
    # Extract d and d_prime
    print("mixed best: ", re.findall(r"mixed_best\(([^,]+),([^,]+),.*?\)\s", model_after_switch.all_atoms))
    print(f"empty_flights: ", re.findall(r"wasted\([^)]+\)", model_after_switch.all_atoms))

    answer_set = model_after_switch.flight_path_fact_format
    print(f"flight path after swap: \n{answer_set}")

    print(f"solution time: ", re.findall(r"solution_time\([^)]+\)", model_after_switch.all_atoms))
        
    print("========== step 3: assign time using clingo-dl")
    # Final scheduling with time constraints
    dl = Schedule(time_limit=30
                    , encoding='encoding/time0.lp'
                    , network=network
                    , heuristic=False
                    , dl_theory=True
                    , max_segment=max_segment
                    )
    time = dl.Solving(fact_load=answer_set)
    if len(time) != 0:
        return 'SAT2'
    else: return 'SAT1'

# Main execution
if __name__ == "__main__":
    s = Schedule(
        time_limit=60,
        update_time_limit=5,
        max_segment=8,
        encoding='T-ITS 2025/schedule.lp',
        network='T-ITS 2025/instances/network_NY_0.lp',
        rq_file='T-ITS 2025/instances/rq.lp',
        init_file='T-ITS 2025/instances/init.lp',
        choose_heu=[2],
        choose_opt=None,
        dl_theory=True,
        horizon=180,
        heuristic=True
    )
    s.Solving()
    print('test')
