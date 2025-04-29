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
    def __init__(self, time_limit: int = 30, start_segment: int = 1, max_segment: int = 13, horizon: int = 180, 
                 encoding: str = 'schedule.lp', network: str = None, heuristic: bool = False, 
                 choose_heu: list = None, choose_opt: list = None, n_rq: int = None, n_agents: int = None, 
                 seed_init: int = None, seed_rq: int = 1, vert_cap: int = 12, dl_theory: bool = False):
        self.thy = ClingoDLTheory()
        self.time_limit = time_limit
        self.start_segment = start_segment
        self.max_segment = max_segment
        self.horizon = horizon
        self.encoding = encoding
        self.network = network
        self.seed_init = seed_init
        self.seed_rq = seed_rq
        self.control_par = ['-c', f'start_seg={self.start_segment}', '-t4']
        if horizon is not None:
            self.control_par.extend(['-c', f'horizon={self.horizon}'])
        if max_segment is not None:
            self.control_par.extend(['-c', f'max_seg={self.max_segment}'])
        if heuristic:
            self.control_par.append('--heuristic=Domain')
        if n_agents is not None:
            InitLoc(nAgents=n_agents, seed=seed_init, limit_nAgents_each_vertiport=vert_cap)
        if n_rq is not None:
            GenRq(cust=n_rq, seed=seed_rq)
        self.choose_heu = choose_heu
        self.choose_opt = choose_opt
        self.dl_theory = dl_theory

    def Solving(self, fact_load=""):
        self.models = []
        ctl = Control(self.control_par)
        self.thy.register(ctl)
        with open(self.encoding, 'r') as file:
            prg = file.read()
        if self.network:
            with open(self.network, 'r') as file:
                prg += file.read()
        if self.choose_opt:
            for opt in self.choose_opt:
                with open(f'opt_heu/opt_{opt}.lp', 'r') as file:
                    prg += file.read()
        if self.choose_heu:
            for heu in self.choose_heu:
                with open(f'opt_heu/heu_{heu}.lp', 'r') as file:
                    prg += file.read()
        prg = fact_load + prg
        with ProgramBuilder(ctl) as bld:
            parse_string(prg, lambda ast: self.thy.rewrite_ast(ast, bld.add))

        try:
            print('Start Grounding...')
            ctl.ground([('base', [])])
            self.thy.prepare(ctl)
            print('Done grounding!')
            print('Start Solving...')
            with ctl.solve(yield_=True, on_model=self.thy.on_model, async_=True) as hnd:
                start_time = time.time()
                time_limit = self.time_limit
                while True:
                    hnd.resume()
                    flag = hnd.wait(time_limit)
                    time_limit = self.time_limit - (time.time() - start_time)
                    print(f"{self.time_limit - time_limit}, {hnd.get()}")
                    if time_limit <= 0:
                        print('Stop solving')
                        hnd.cancel()
                        break
                    self.models.append(self.make_model(hnd.model()))
                    print(LINE)
                hnd.cancel()
        except KeyboardInterrupt:
            print("\nSolving interrupted by keyboard...")
        finally:
            return self.models

    def make_model(self, rawmodel: Model):
        model = argparse.Namespace()
        model.opt_cost = rawmodel.cost.copy()
        model.number = rawmodel.number
        model.output = str(rawmodel)
        model.all_atoms = " ".join(map(str, rawmodel.symbols(atoms=True)))
        model.flight_path = [symbol for symbol in str(rawmodel).split() if symbol.startswith('as')]
        model.flight_path_fact_format = " ".join([i + '.' for i in model.flight_path])
        if self.dl_theory:
            dl_assignments = [f"dl({str(dl_variable)},{dl_value})"
                              for dl_variable, dl_value in self.thy.assignment(rawmodel.thread_id)]
            model.dl_assignments = dl_assignments
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

# Switching trajectory approach 1
def SwitchingTrajectory():
    # Initial solution
    print('=========== step 1: assign trajectory allowing operation time > horizon, prioritizing serving all customers:')
    s = Schedule(time_limit=30
                 , start_segment=1
                 , max_segment=13
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
    print(answer_set)
    print("========== step 2: start switching trajectory to constraint operation time <= horizon")
    while stop == False:
        sw = Schedule(time_limit=8
                      , encoding='encoding/swap0.1.lp'
                      , network='instances/network_NY_0.lp'
                      , horizon=180
                      , heuristic=False
                      , choose_heu=None
                      , choose_opt=None
                      , dl_theory=False
                      )
        model = sw.Solving(fact_load=answer_set)
        model_after_switch = model[0]
        # Extract d and d_prime
        print("mixed best: ", re.findall(r"mixed_best\(([^,]+),([^,]+),.*?\)\s", model_after_switch.all_atoms))
        print(f"empty_flights: ", re.findall(r"wasted\([^)]+\)", model_after_switch.all_atoms))

        answer_set = model_after_switch.flight_path_fact_format
        print(f"flight path after swap: \n{answer_set}")

        print(f"solution time: ", re.findall(r"solution_time\([^)]+\)", model_after_switch.all_atoms))
        stop = True
    print("========== step 3: assign time using clingo-dl")
    # Final scheduling with time constraints
    dl = Schedule(time_limit=30
                  , encoding='encoding/time0.lp'
                  , network='instances/network_NY_0.lp'
                  , heuristic=False
                  , dl_theory=True
                  , horizon=None
                  , max_segment=13
                  )
    time = dl.Solving(fact_load=answer_set)
    return time[0].to_visualized

# Main execution
if __name__ == "__main__":
    time = SwitchingTrajectory()
    print(time)
    # with open("results/trajectories.lp", "w") as file:
    #     file.write(time)