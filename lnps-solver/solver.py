"""
Solver
@author Irumi Sugimori
Time-stamp: <2024-04-06 15:20:30 irumisugimori>
"""

import signal
import sys
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod
from enum import IntEnum

from clingo.application import Flag
from clingo.control import Control


class LogLevel(IntEnum):
    NONE = 0
    BASIC = 1
    MORE = 2
    FULL = 3

    
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]


def print_message(*args, file=sys.stdout):
    elems = [str(x) for x in args]
    print(" ".join(elems), file=file, flush=True)
    

def print_comment(*args):
    print_message("c", *args)
    

def print_answer(*args):
    print_message("a", *args)
    

def print_solution(*args):
    print_message("s", *args)

    
def print_variable(*args):
    print_message("v", *args)
    

def print_warning(*args):
    print_message("w", *args, file=sys.stderr)

    
def set_signal_handler(handler):
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGALRM, handler)
    signal.signal(signal.SIGTERM, handler)


class HeulingoConfig:
    heulingo_configuration_values: dict
    iter_configuration: Optional[str]
    iter_opt_strategy: Optional[str]
    iter_opt_heuristic: Optional[str]
    iter_restart_on_model: Flag
    has_iter_restart_on_model: bool
    iter_heuristic: str
    iter_opt_mode: dict
    iter_solve_limit: Optional[str]
    solve_limit_increase_rate: float
    acceptance_rate: float
    random_seed: Optional[int]
    heulingo_configuration: Optional[str]
    log_level: int

    heulingo_configuration_values = {'teaspoon': {'configuration': 'jumpy',
                                                  'opt-strategy': 'usc,11',
                                                  'parallel-mode': None,
                                                  'solve-limit': "2500000,5000",
                                                  'iter-configuration': "tweety",
                                                  'iter-opt-strategy': "bb,0",
                                                  'iter-opt-heuristic': "3",
                                                  'iter-restart-on-model': "1",
                                                  'iter-opt-mode': None,
                                                  'iter-solve-limit': "30000"},
                                     'tsp': {'configuration': None,
                                             'opt-strategy': None,
                                             'parallel-mode': None,
                                             'solve-limit': "1210000",
                                             'iter-configuration': None,
                                             'iter-opt-strategy': None,
                                             'iter-opt-heuristic': None,
                                             'iter-restart-on-model': None,
                                             'iter-opt-mode': None,
                                             'iter-solve-limit': "800000"},
                                     'sgp': {'configuration': None,
                                             'opt-strategy': None,
                                             'parallel-mode': None,
                                             'solve-limit': "500000",
                                             'iter-configuration': None,
                                             'iter-opt-strategy': None,
                                             'iter-opt-heuristic': None,
                                             'iter-restart-on-model': None,
                                             'iter-opt-mode': "opt,0,dynamic",
                                             'iter-solve-limit': "500000"},
                                     'spg': {'configuration': "many",
                                             'opt-strategy': None,
                                             'parallel-mode': "4",
                                             'solve-limit': "300000",
                                             'iter-configuration': None,
                                             'iter-opt-strategy': None,
                                             'iter-opt-heuristic': None,
                                             'iter-restart-on-model': None,
                                             'iter-opt-mode': None,
                                             'iter-solve-limit': "6000"},
                                     'wsc,medium': {'configuration': None,
                                                    'opt-strategy': "usc,15",
                                                    'parallel-mode': None,
                                                    'solve-limit': "1000000",
                                                    'iter-configuration': None,
                                                    'iter-opt-strategy': "bb,0",
                                                    'iter-opt-heuristic': None,
                                                    'iter-restart-on-model': None,
                                                    'iter-opt-mode': None,
                                                    'iter-solve-limit': "40000"},
                                     'wsc,large': {'configuration': None,
                                                    'opt-strategy': "usc,15",
                                                    'parallel-mode': None,
                                                    'solve-limit': "30000000",
                                                    'iter-configuration': None,
                                                    'iter-opt-strategy': "bb,0",
                                                    'iter-opt-heuristic': None,
                                                    'iter-restart-on-model': None,
                                                    'iter-opt-mode': None,
                                                    'iter-solve-limit': "60000"},
                                     'sd': {'configuration': "handy",
                                            'opt-strategy': "usc,3",
                                            'parallel-mode': None,
                                            'solve-limit': "900000",
                                            'iter-configuration': None,
                                            'iter-opt-strategy': None,
                                            'iter-opt-heuristic': None,
                                            'iter-restart-on-model': None,
                                            'iter-opt-mode': "opt,0,dynamic",
                                            'iter-solve-limit': "40000"}
                                     }

    def __init__(self):
        self.iter_configuration = None
        self.iter_opt_strategy = None
        self.iter_opt_heuristic = None
        self.iter_restart_on_model = Flag()
        self.has_iter_restart_on_model = False
        self.iter_heuristic = "Domain"
        self.iter_opt_mode = {'mode': None, 'nf': None, 'modifier': None}
        self.iter_solve_limit = None
        self.solve_limit_increase_rate = 0.01
        self.acceptance_rate = 0
        self.random_seed = 0
        self.heulingo_configuration = None
        self.log_level = LogLevel.NONE

        
class SolverConfig:
    configuration: Optional[str]
    opt_strategy: Optional[str]
    opt_heuristic: Optional[str]
    restart_on_model: Optional[str]
    heuristic: Optional[str]
    opt_mode: Optional[str]
    solve_limit: Optional[str]
    
    def __init__(self):
        self.configuration = None
        self.opt_strategy = None
        self.opt_heuristic = None
        self.restart_on_model = None
        self.heuristic = None
        self.opt_mode = None
        self.solve_limit = None
        
    
class Solver(ABC):
    __model_last: Optional[str]
    __cost_last: list[Optional[int]]
    __variability: Optional[bool]
    _heulingo_config: HeulingoConfig
    _ctl: Control
    _finished: bool
    _result: str
    _optimum: str
    
    def __init__(self, ctl: Control, heulingo_config: HeulingoConfig):
        self.__model_last = None
        self.__cost_last = [None]
        self.__variability = None
        self._heulingo_config = heulingo_config
        self._ctl = ctl
        self._finished = False
        self._result = "UNKNOWN"
        self._optimum = "unknown"
        set_signal_handler(self.__handler)

    def __handler(self, dummy_a, dummy_b):
        self._finished = True
        self._ctl.interrupt()

    def __on_model(self, model):
        self.__model_last = str(model)
        self.__cost_last = model.cost
        
    def _print_debug(self, *args, log_level: LogLevel = LogLevel.BASIC):
        if self._heulingo_config.log_level >= log_level:
            print_message("d", f"{log_level}", *args)

    def _on_finish(self, ret):
        if ret.satisfiable:
            self._result = "SATISFIABLE"
            if not self.__cost_last:
                self._finished = True
                return
            
        if self.__variability and ret.exhausted:
            if ret.unsatisfiable:
                self._result = "UNSATISFIABLE"
            else:
                self._result = "OPTIMUM FOUND"
                self._optimum = "yes"
            self._finished = True
        
    def _find(self, config: Optional[SolverConfig] = None, variability: bool = True) -> Tuple[Optional[str], list[Optional[int]]]:
        if config is not None:
            if config.configuration is not None:
                self._ctl.configuration.configuration = config.configuration
            if config.opt_strategy is not None:
                self._ctl.configuration.solver.opt_strategy = config.opt_strategy
            if config.opt_heuristic is not None:
                self._ctl.configuration.solver.opt_heuristic = config.opt_heuristic
            if config.restart_on_model is not None:
                self._ctl.configuration.solver.restart_on_model = config.restart_on_model
            if config.heuristic is not None:
                self._ctl.configuration.solver.heuristic = config.heuristic
            if config.opt_mode is not None:
                self._ctl.configuration.solve.opt_mode = config.opt_mode
            if config.solve_limit is not None:
                self._ctl.configuration.solve.solve_limit = config.solve_limit

        self._print_debug("configuration:", self._ctl.configuration.configuration)
        self._print_debug("opt-strategy:", self._ctl.configuration.solver.opt_strategy)
        self._print_debug("parallel-mode:", self._ctl.configuration.solve.parallel_mode)
        self._print_debug("opt-heuristic:", self._ctl.configuration.solver.opt_heuristic)
        self._print_debug("restart-on-model:", self._ctl.configuration.solver.restart_on_model)
        self._print_debug("heuristic:", self._ctl.configuration.solver.heuristic)
        self._print_debug("opt-mode:", self._ctl.configuration.solve.opt_mode)
        self._print_debug("solve-limit:", self._ctl.configuration.solve.solve_limit)
        
        self.__variability = variability
            
        handle = self._ctl.solve(on_model=self.__on_model,
                                 on_finish=self._on_finish,
                                 async_=True)
        while not handle.wait(0):
            pass

        if self.__model_last is None and not self._finished:
            print_warning(f"First solution found is used as initial solution because {self._ctl.configuration.solve.solve_limit} of solve-limit is not enough to find solution")
            solve_limit_tmp = self._ctl.configuration.solve.solve_limit
            models_tmp = self._ctl.configuration.solve.models
            self._ctl.configuration.solve.solve_limit = "umax"
            self._ctl.configuration.solve.models = 1
            self._print_debug("solve-limit:", self._ctl.configuration.solve.solve_limit)
            self._print_debug("models:", self._ctl.configuration.solve.models)
            handle = self._ctl.solve(on_model=self.__on_model,
                                     on_finish=self._on_finish,
                                     async_=True)
            while not handle.wait(0):
                pass
            self._ctl.configuration.solve.solve_limit = solve_limit_tmp
            self._ctl.configuration.solve.models = models_tmp
        
        return self.__model_last, self.__cost_last

    @abstractmethod
    def solve(self):
        pass
