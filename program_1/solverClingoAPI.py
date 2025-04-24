from clingo import Control
from clingo.ast import parse_string, ProgramBuilder
from clingodl import ClingoDLTheory
import sys
import argparse
import os
import time
import subprocess
from utils.utils import CalProfit, EstimateMinAgents
import random
from typing import List

def on_model(m):
    print (m)
LINE = "---------------------------------------------------------------------"
# Argument parsing
parser = argparse.ArgumentParser(description='Solver for eVTOL scheduling.')
parser.add_argument('--n_rq', type=int, default = 30, help='Number of requests')
parser.add_argument('--n_agents', type=int, default = 32, help='Number of agents')
parser.add_argument('--max_segment', type=int, default = 11, help='Maximum segment')
parser.add_argument('--horizon', type=int, default = 180, help='Horizon')
parser.add_argument('--time_limit', type=int, default = 30, help='Time limit')
parser.add_argument('--seed', type=int, default = 1, help='seed for random initial location')
parser.add_argument('--vertiport_cap', type=int, default = 6, help='vertiport capacity')
args = parser.parse_args()

min_agents, min_segments=EstimateMinAgents(horizon=args.horizon, b_init=60, demand_cust=args.n_rq, aircraft_capacity=4)

def InitNagent(nagents):
    if nagents is not None:
        gen_init = ["python"
                    ,"instances/gen_init_random_NY.py"
                    , str(nagents)
                    , str(args.seed)
                    , str(args.vertiport_cap)
                    ]
        subprocess.run(gen_init)  # Run gen_init script

def InitRq(n_rq):
    if n_rq is not None:
        gen_rq = ["python"
                , "instances/gen_rq_NY.py"
                , str(n_rq)
                ]
        subprocess.run(gen_rq)  # Run gen_rq script
    

class Schedule:
    def __init__(self
                 , time_limit = args.time_limit
                 , start_segment = 1
                 , max_segment = args.max_segment
                 , horizon = args.horizon
                 , encoding = 'schedule.lp'
                 , heuristic= True
                 , choose_heu = [2]
                 , choose_opt = [2]
                 , n_rq = args.n_rq
                 , n_agents = args.n_agents
                 ):
        # initialize clingo-dl theory
        self.thy = ClingoDLTheory()
        self.time_limit = time_limit
        self.start_segment = start_segment
        self.max_segment = max_segment
        self.horizon = horizon
        self.encoding = encoding
        self.control_par =  ['-c', f'start_seg={self.start_segment}'
                            , '-c', f'max_seg={self.max_segment}'
                            , '-c', f'horizon={self.horizon}'
                            , '-t4'
                            ]
        self.choose_heu = choose_heu
        self.choose_opt = choose_opt
        if heuristic == True:
            self.control_par.append('--heuristic=Domain')
        InitNagent(nagents=n_agents)
        InitRq(n_rq=n_rq)

    def Solving(self, fact_load = ""):
        self.models = []

        # init control object that control the grounding and solving
        ctl = Control(self.control_par)
        self.thy.register(ctl)
        
        # Read schedule.lp content into variable prg
        with open(self.encoding, 'r') as file:
            prg = file.read()
            
        # add opt specfication to program
        for opt in self.choose_opt:
            with open(f'opt_heu/opt_{opt}.lp', 'r') as file:
                prg += file.read()
        # add heu spec to program
        for heu in self.choose_heu:
            with open(f'opt_heu/heu_{heu}.lp', 'r') as file:
                prg += file.read()        
        # add addtional facts to program
        prg=fact_load+prg
        # Write into control object
        with ProgramBuilder(ctl) as bld:
            parse_string(prg, lambda ast: self.thy.rewrite_ast(ast, bld.add))

        try:
            # grounding
            print('Start Grounding...')
            ctl.ground([('base', [])])
            self.thy.prepare(ctl)
            print('Done grounding!')
            # solving schedule.lp
            print('Start Solving...')
            with ctl.solve(yield_=True, on_model = self.thy.on_model, async_=True) as hnd:
                start_time = time.time()
                def time_left(): return (time.time() - start_time)
                time_limit = self.time_limit
                # loop that wait for signal from solveHandle hnd
                while True:
                    hnd.resume()
                    flag = hnd.wait(time_limit) # hnd.wait return True if there is model, return False if there isn't model after time_limit solving
                    time_limit = self.time_limit - time_left()
                    print(f"{self.time_limit-time_limit}, {hnd.get()}")
                    if time_limit <= 0: # end of solving limit time
                        print('Stop solving')
                        hnd.cancel()
                        break
                    
                    self.models.append(self.make_model(hnd.model()))
                        

                        
                    # print(m)
                    print(LINE)
                hnd.cancel()
        except KeyboardInterrupt:
            print("\nSolving interrupted by keyboard...")
        finally:
            return self.models

    
    def make_model(self,rawmodel):
        model = argparse.Namespace()
        model.opt_cost = rawmodel.cost.copy() # optimization value of answer set
        model.number = rawmodel.number
        # model.optimality_proven = rawmodel.optimality_proven
        # model.thread_id = rawmodel.thread_id
        # model.type = rawmodel.type
        # model.symbols = list(rawmodel.symbols(
        #     atoms=True, terms=True, theory=True)).copy()
        # model.shown = list(rawmodel.symbols(shown=True)).copy()
        model.flight_path = str(rawmodel) # atom as
        model.flight_path_clingo_format = " ".join([i + '.' for i in model.flight_path.split()])
        
        dl_assignments = []
        for dl_variable, dl_value in self.thy.assignment(rawmodel.thread_id):
            dl_assignments.append(f"dl({str(dl_variable)},{dl_value})")
        model.dl_assignments = dl_assignments # atom dl
        # calculate revenue, cost, profit
        # model.revenue, model.em_cost, model.chg_cost, model.profit = CalProfit(model.flight_path.split() + model.dl_assignments)
        ComputeRevenue(model.flight_path)
        model.to_visualized = model.flight_path + " "+ " ".join(model.dl_assignments)
        GetNumberOfAgentsEachVertiport(model.dl_assignments)
        print(model.to_visualized)
        return model
    
def GetBestModelOptCost(models:List[argparse.Namespace]):
    if not models:
        return None
    # Return the model with the highest profit
    return max(models, key=lambda model: model.opt_cost)

def GetBestModelProfit(models:List[argparse.Namespace]):
    if not models:
        return None
    # Return the model with the highest profit
    return max(models, key=lambda model: model.profit)

def ComputeRevenue(flight_path):
    flight_path = " ".join([i + '.' for i in flight_path.split()])
    # print(flight_path)
    ctl = Control([])
    ctl.add(flight_path)
    ctl.load('compute_revenue.lp')
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as out:
        print(out.model())

def GetNumberOfAgentsEachVertiport(dl_assignments):
    schedule = " ".join([i + '.' for i in dl_assignments])
    ctl = Control([])
    ctl.add(schedule)
    ctl.load('transfer.lp')
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as out:
        print(out.model())
    
# processing model resulting from solver

def CentralSchedule():
    schedule = Schedule()
    models = schedule.Solving()
    best_model = GetBestModelProfit(models)
    return best_model

def ConsecutiveSchedule():
    schedule = Schedule(choose_opt=[2,2.1])
    models = schedule.Solving()
    best_model = GetBestModelOptCost(models)
    flight_path = best_model.flight_path_clingo_format
    
    schedule = Schedule(start_segment=11, max_segment=13, heuristic=False, choose_opt=[1,2.1])
    print(flight_path)
    models = schedule.Solving(fact_load=flight_path)
    best_model = GetBestModelOptCost(models)

    return best_model

def DecoupledSchedule():
    flight_path =''
    for d in range(0, args.n_agents):
        
        schedule = Schedule(encoding='decoupled_scheduling.lp')
        models = schedule.Solving(fact_load=f'agent({d}).'+ flight_path)
        flight_path += models[-1].flight_path_clingo_format
    for d in range(0, args.n_agents):
        schedule = Schedule(encoding='decoupled_scheduling.lp', start_segment=11, max_segment=14, heuristic=False, choose_opt=[1,2.1])
        models = schedule.Solving(fact_load=f'agent({d}).'+ flight_path)
        flight_path += models[-1].flight_path_clingo_format
    return models[-1]
    
def VertiportConstraintSchedule():
    schedule= Schedule(encoding='schedule_vertiport_constraint.lp', max_segment=7, n_agents=10)
    models = schedule.Solving()
    best_model = GetBestModelProfit(models)
    return best_model

class IncrementalSchedule(Schedule):
    def Solving(self, fact_load=""):
        self.models = []

        # init control object that control the grounding and solving
        ctl = Control(self.control_par)
        self.thy.register(ctl)
        
        # Read schedule.lp content into variable prg
        with open(self.encoding, 'r') as file:
            prg = file.read()
            
        # add opt specfication to program
        for opt in self.choose_opt:
            with open(f'opt_heu/opt_{opt}.lp', 'r') as file:
                prg += file.read()
        # add heu spec to program
        for heu in self.choose_heu:
            with open(f'opt_heu/heu_{heu}.lp', 'r') as file:
                prg += file.read()        
        # add addtional facts to program
        prg=fact_load+prg
        # Write into control object
        with ProgramBuilder(ctl) as bld:
            parse_string(prg, lambda ast: self.thy.rewrite_ast(ast, bld.add))

        # grounding
        ctl.ground([('base', []), ('step',[0])])
        self.thy.prepare(ctl)
        print('Done grounding')
        # solving schedule.lp
        try:
            with ctl.solve(yield_=True, on_model = self.thy.on_model, async_=True) as hnd:
                start_time = time.time()
                def time_left(): return (time.time() - start_time)
                time_limit = self.time_limit
                # loop that wait for signal from solveHandle hnd
                while True:
                    hnd.resume()
                    flag = hnd.wait(time_limit) # hnd.wait return True if there is model, return False if there isn't model after time_limit solving
                    time_limit = self.time_limit - time_left()
                    print(f"{self.time_limit-time_limit}, {flag}")
                    if time_limit >= self.time_limit or flag == False: # end of solving limit time
                        print('Stop solving')
                        break
                    
                    self.models.append(self.make_model(hnd.model()))
                        

                        
                    # print(m)
                    print(LINE)
                hnd.cancel()
        except KeyboardInterrupt:
            print("\nSolving interrupted by keyboard...")
        finally:
            return self.models



# print(f'number of models:{len(models)}')


# print model


if __name__ == "__main__":
    InitNagent(args.n_agents)
    InitRq(args.n_rq)
    pass
    # model = ConsecutiveSchedule()
    # with open("results/trajectories.lp", "w") as file:
    #     file.write(model.to_visualized)