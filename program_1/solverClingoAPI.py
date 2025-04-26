from clingo import Control
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

# Callback function to handle models during solving
def on_model(m):
    print (m)

# Separator line for output readability
LINE = "---------------------------------------------------------------------"

# Uncommented argument parsing section for potential CLI usage
# parser = argparse.ArgumentParser(description='Solver for eVTOL scheduling.')
# parser.add_argument('--n_rq', type=int, default = 30, help='Number of requests')
# parser.add_argument('--n_agents', type=int, default = 34, help='Number of agents')
# parser.add_argument('--max_segment', type=int, default = 11, help='Maximum segment')
# parser.add_argument('--horizon', type=int, default = 180, help='Horizon')
# parser.add_argument('--time_limit', type=int, default = 30, help='Time limit')
# parser.add_argument('--seed', type=int, default = 1, help='seed for random initial location')
# parser.add_argument('--vertiport_cap', type=int, default = 6, help='vertiport capacity')
# args = parser.parse_args()

# Estimate the minimum number of agents and segments required
min_agents, min_segments = EstimateMinAgents(horizon=180, b_init=60, demand_cust=30, aircraft_capacity=4)

# Class to handle scheduling logic
class Schedule:
    def __init__(self
                 , time_limit = 30  # Time limit for solving
                 , start_segment = 1  # Starting segment
                 , max_segment = 13  # Maximum segment
                 , horizon = 180  # Time horizon
                 , encoding = 'schedule.lp'  # Encoding file
                 , network = None  # Network file (optional)
                 , heuristic= False  # Use heuristic or not
                 , choose_heu = None  # Heuristic options
                 , choose_opt = None  # Optimization options
                 , n_rq = None  # Number of requests
                 , n_agents = None  # Number of agents
                 , seed_init = None  # Seed for initial location
                 , seed_rq = 1  # Seed for request generation
                 , vert_cap = 12  # Vertiport capacity
                 , dl_theory = False  # Use ClingoDL theory or not
                 ):
        # Initialize ClingoDL theory
        self.thy = ClingoDLTheory()
        self.time_limit = time_limit
        self.start_segment = start_segment
        self.max_segment = max_segment
        self.horizon = horizon
        self.encoding = encoding
        self.network = network
        self.seed_init = seed_init
        self.seed_rq = seed_rq
        # Control parameters for Clingo
        self.control_par = ['-c', f'start_seg={self.start_segment}', '-t4']
        if horizon != None:
            self.control_par.extend(['-c', f'horizon={self.horizon}'])
        if max_segment != None:
            self.control_par.extend(['-c', f'max_seg={self.max_segment}'])
            
        self.choose_heu = choose_heu
        self.choose_opt = choose_opt
        self.dl_theory = dl_theory
        # Add heuristic option if enabled
        if heuristic == True:
            self.control_par.append('--heuristic=Domain')
        # Initialize locations if number of agents is provided
        if n_agents != None:
            InitLoc(nAgents=n_agents, seed=seed_init, limit_nAgents_each_vertiport=vert_cap)
        # Generate requests if number of requests is provided
        if n_rq != None:
            GenRq(cust=n_rq, seed=seed_rq)

    # Method to solve the scheduling problem
    def Solving(self, fact_load = ""):
        self.models = []

        # Initialize Clingo control object
        ctl = Control(self.control_par)
        self.thy.register(ctl)
        
        # Read the main encoding file
        with open(self.encoding, 'r') as file:
            prg = file.read()
        # Append network file content if provided
        if self.network != None:
            with open(self.network, 'r') as file:
                prg += file.read()
        # Append optimization specifications if provided
        if self.choose_opt != None:
            for opt in self.choose_opt:
                with open(f'opt_heu/opt_{opt}.lp', 'r') as file:
                    prg += file.read()
        # Append heuristic specifications if provided
        if self.choose_heu != None:
            for heu in self.choose_heu:
                with open(f'opt_heu/heu_{heu}.lp', 'r') as file:
                    prg += file.read()
        # Add additional facts to the program
        prg = fact_load + prg
        # Write the program into the control object
        with ProgramBuilder(ctl) as bld:
            parse_string(prg, lambda ast: self.thy.rewrite_ast(ast, bld.add))

        try:
            # Grounding the program
            print('Start Grounding...')
            ctl.ground([('base', [])])
            self.thy.prepare(ctl)
            print('Done grounding!')
            # Solving the program
            print('Start Solving...')
            with ctl.solve(yield_=True, on_model=self.thy.on_model, async_=True) as hnd:
                start_time = time.time()
                def time_left(): return (time.time() - start_time)
                time_limit = self.time_limit
                # Loop to wait for solving results
                while True:
                    hnd.resume()
                    flag = hnd.wait(time_limit)  # Wait for a model or timeout
                    time_limit = self.time_limit - time_left()
                    print(f"{self.time_limit-time_limit}, {hnd.get()}")
                    if time_limit <= 0:  # Stop solving if time limit is reached
                        print('Stop solving')
                        hnd.cancel()
                        break

                    # Append the model to the list of models
                    self.models.append(self.make_model(hnd.model()))
                        
                    print(LINE)
                hnd.cancel()
        except KeyboardInterrupt:
            print("\nSolving interrupted by keyboard...")
        finally:
            return self.models

    # Method to process raw models into a structured format
    def make_model(self, rawmodel):
        model = argparse.Namespace()
        model.opt_cost = rawmodel.cost.copy()  # Optimization cost of the answer set
        model.number = rawmodel.number
        model.output = str(rawmodel)  # Raw output of the model
        # Extract flight paths from the model
        model.flight_path = [symbol for symbol in str(rawmodel).split() if symbol.startswith('as')]
        model.flight_path_fact_format = " ".join([i + '.' for i in model.flight_path])
        if self.dl_theory == True:
            dl_assignments = []
            for dl_variable, dl_value in self.thy.assignment(rawmodel.thread_id):
                dl_assignments.append(f"dl({str(dl_variable)},{dl_value})")
            model.dl_assignments = dl_assignments  # DL assignments
            # Prepare data for visualization
            model.to_visualized = model.flight_path_fact_format + " ".join([i + '.' for i in model.dl_assignments])

        return model

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

# Vertiport constraint scheduling approach
def VertiportConstraintSchedule():
    schedule = Schedule(encoding='schedule_vertiport_constraint.lp', max_segment=7, n_agents=10)
    models = schedule.Solving()
    best_model = GetBestModelProfit(models)
    return best_model

# Switching trajectory approach
def SwitchingTrajectory():
    # Initial solution
    s = Schedule(time_limit=30
                 , start_segment=1
                 , max_segment=13
                 , horizon=180
                 , encoding='encoding/s.lp'
                 , network='instances/network_NY_0.lp'
                 , heuristic=True
                 , choose_heu=None
                 , choose_opt=None
                 , n_rq=30
                 , n_agents=34
                 , seed_init=1
                 , seed_rq=1
                 , dl_theory=False
                 )
    models = s.Solving()
    best_model = GetBestModelOptCost(models=models)
    stop = False
    # Switching logic
    answer_set = best_model.flight_path_fact_format
    print("start switching!")
    previous_min_time = ""
    previous_max_time = ""
    while stop == False:
        sw = Schedule(time_limit=30
                      , encoding='encoding/solution.lp'
                      , network='instances/network_NY_0.lp'
                      , heuristic=False
                      , choose_heu=None
                      , choose_opt=None
                      )
        model = sw.Solving(fact_load=answer_set)
        model_after_switch = model[0]
        # Extract min_time and max_time
        min_time = int(re.search(r"min_time\((\d+)\)", model_after_switch.output).group(1))
        max_time = int(re.search(r"max_time\((\d+)\)", model_after_switch.output).group(1))
        print(f"previous min time agent {previous_min_time} and previous max time agent {previous_max_time}")
        if (min_time, max_time) == (previous_min_time, previous_max_time) or (max_time, min_time) == (previous_min_time, previous_max_time):
            stop = True
        print(f"min time agent {min_time} and max time agent {max_time}")
        print("flight path before swap:")
        pattern_min_time = re.compile(rf"as\({str(min_time)},")
        pattern_max_time = re.compile(rf"as\({str(max_time)},")
        print(ClingoSolver(prg_path="sort_traj.lp", facts=" ".join([f for f in answer_set.split() if pattern_min_time.match(f)])))
        print(ClingoSolver(prg_path="sort_traj.lp", facts=" ".join([f for f in answer_set.split() if pattern_max_time.match(f)])))
        print(f"best cuts: ", re.findall(r"mixed_best\([^)]+\)", model_after_switch.output))
        
        previous_min_time = min_time
        previous_max_time = max_time
        answer_set = model_after_switch.flight_path_fact_format
        print("flight path after swap:")
        print(ClingoSolver(prg_path="sort_traj.lp", facts=" ".join([f for f in answer_set.split() if pattern_min_time.match(f)])))
        print(ClingoSolver(prg_path="sort_traj.lp", facts=" ".join([f for f in answer_set.split() if pattern_max_time.match(f)])))
        print(f"solution time: ", re.findall(r"solution_time\([^)]+\)", model_after_switch.output))
    print(f"total time needed: ", re.findall(r"total_time_need\([^)]+\)", model_after_switch.output))

    # Final scheduling with time constraints
    dl = Schedule(time_limit=30
                  , encoding='encoding/time_0.lp'
                  , network='instances/network_NY_0.lp'
                  , heuristic=False
                  , dl_theory=True
                  , horizon=None
                  , max_segment=None
                  )
    time = dl.Solving(fact_load=answer_set)
    return time[0].to_visualized

# Main execution
if __name__ == "__main__":
    time = SwitchingTrajectory()
    with open("results/trajectories.lp", "w") as file:
        file.write(time)