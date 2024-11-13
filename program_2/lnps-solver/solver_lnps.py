"""
Solver LNPS
@author Irumi Sugimori
Time-stamp: <2024-04-06 15:21:34 irumisugimori>
"""

import math
import random
from typing import Optional

from clingo.control import Control
from clingo.symbol import Symbol, SymbolType, Function, Number, parse_term

from solver import *


UINT_MAX = 4294967295
LINE = "--------------------------------------------------------------------------------------"


class SolverLNPS(Solver):
    def __init__(self, ctl: Control, heulingo_config: HeulingoConfig):
        super().__init__(ctl, heulingo_config)
        self.__lnps_config = []
        
        self.__iter_solver_config = SolverConfig()

        self.__iter_solver_config.solve_limit = self._ctl.configuration.solve.solve_limit
        
        if self._heulingo_config.heulingo_configuration is not None:
            self.__iter_solver_config.configuration = HeulingoConfig.heulingo_configuration_values[self._heulingo_config.heulingo_configuration]['iter-configuration']
            self.__iter_solver_config.opt_strategy = HeulingoConfig.heulingo_configuration_values[self._heulingo_config.heulingo_configuration]['iter-opt-strategy']
            self.__iter_solver_config.opt_heuristic = HeulingoConfig.heulingo_configuration_values[self._heulingo_config.heulingo_configuration]['iter-opt-heuristic']
            self.__iter_solver_config.restart_on_model = HeulingoConfig.heulingo_configuration_values[self._heulingo_config.heulingo_configuration]['iter-restart-on-model']
            self.__iter_solver_config.solve_limit = HeulingoConfig.heulingo_configuration_values[self._heulingo_config.heulingo_configuration]['iter-solve-limit']
            
        if self._heulingo_config.iter_configuration is not None:
            self.__iter_solver_config.configuration = self._heulingo_config.iter_configuration
        if self._heulingo_config.iter_opt_strategy is not None:
            self.__iter_solver_config.opt_strategy = self._heulingo_config.iter_opt_strategy
        if self._heulingo_config.iter_opt_heuristic is not None:
            self.__iter_solver_config.opt_heuristic = self._heulingo_config.iter_opt_heuristic
        if self._heulingo_config.has_iter_restart_on_model:
            self.__iter_solver_config.restart_on_model = str(int(bool(self._heulingo_config.iter_restart_on_model)))
        self.__iter_solver_config.heuristic = self._heulingo_config.iter_heuristic
        if self._heulingo_config.iter_solve_limit is not None:
            self.__iter_solver_config.solve_limit = self._heulingo_config.iter_solve_limit

        self._print_debug(LINE, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.configuration:", self.__iter_solver_config.configuration, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.opt_strategy:", self.__iter_solver_config.opt_strategy, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.opt_heuristic:", self.__iter_solver_config.opt_heuristic, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.restart_on_model:", self.__iter_solver_config.restart_on_model, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.heuristic:", self.__iter_solver_config.heuristic, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.opt_mode:", self.__iter_solver_config.opt_mode, log_level=LogLevel.FULL)
        self._print_debug("iter_solver_config.solve_limit:", self.__iter_solver_config.solve_limit, log_level=LogLevel.FULL)

        random.seed(self._heulingo_config.random_seed)

    def __increase_solve_limit(self, solver_config: SolverConfig):
        increased_solve_limit = []
        for n in solver_config.solve_limit.split(","):
            if n == "umax":
                increased_solve_limit.append(n)
            else:
                new_n = math.ceil(int(n) * self._heulingo_config.solve_limit_increase_rate / 100 + int(n))
                if new_n <= UINT_MAX:
                    increased_solve_limit.append(str(new_n))                    
                else:
                    increased_solve_limit.append("umax")

        solver_config.solve_limit =  ",".join(increased_solve_limit)

    def __calc_opt_bound(self, solver_config: SolverConfig, cost: list[int]):
        if self._heulingo_config.iter_opt_mode['modifier'] == "static":
            solver_config.opt_mode = self._heulingo_config.iter_opt_mode['mode'] + "," + self._heulingo_config.iter_opt_mode['nf']
        elif self._heulingo_config.iter_opt_mode['modifier'] == "dynamic":
            bounds = cost[:-1]
            bounds.append(math.ceil(cost[-1] + abs(cost[-1]) * self._heulingo_config.iter_opt_mode['nf'] / 100) - 1)
            solver_config.opt_mode = self._heulingo_config.iter_opt_mode['mode'] + "," + (",".join([str(i) for i in bounds]))
        elif self._heulingo_config.iter_opt_mode['mode'] is not None:
            solver_config.opt_mode = self._heulingo_config.iter_opt_mode['mode']

    def __get_lnps_config(self, sol: str):
        for x in self._ctl.symbolic_atoms.by_signature("_lnps_project",2):
            self.__lnps_config.append(dict(predicate_name=x.symbol.arguments[0].name,arity=x.symbol.arguments[1].number))

        if not self.__lnps_config:
            atom_dicts = []
            for atom in sol.split(' '):
                symbol = parse_term(atom)
                atom_dicts.append(dict(predicate_name=symbol.name,arity=len(symbol.arguments)))
            self.__lnps_config = get_unique_list(atom_dicts)

        for i in range(len(self.__lnps_config)):
            for x in self._ctl.symbolic_atoms.by_signature("_lnps_destroy",4):
                if (x.symbol.arguments[0].name == self.__lnps_config[i]['predicate_name']) and (x.symbol.arguments[1].number == self.__lnps_config[i]['arity']):
                    self.__lnps_config[i].setdefault('mask',[]).append(x.symbol.arguments[2].number)
                    self.__lnps_config[i].setdefault('pn',[]).append(x.symbol.arguments[3])
            for x in self._ctl.symbolic_atoms.by_signature("_lnps_prioritize",4):
                if (x.symbol.arguments[0].name == self.__lnps_config[i]['predicate_name']) and (x.symbol.arguments[1].number == self.__lnps_config[i]['arity']):
                    self.__lnps_config[i]['weight'] = x.symbol.arguments[2]
                    self.__lnps_config[i]['modifier'] = x.symbol.arguments[3]
								   
            self.__lnps_config[i].setdefault('mask',[2**self.__lnps_config[i]['arity']-1])
            self.__lnps_config[i].setdefault('pn',[Function('p',[Number(0)])])
            self.__lnps_config[i].setdefault('weight',Number(1))
            self.__lnps_config[i].setdefault('modifier',Function('true'))

        self._print_debug(LINE)
        for conf in self.__lnps_config:
            self._print_debug(f"_lnps_project({conf['predicate_name']},{conf['arity']}).")
            for i in range(len(conf['mask'])):
                self._print_debug(f"_lnps_destroy({conf['predicate_name']},{conf['arity']},{conf['mask'][i]},{conf['pn'][i]}).")
            self._print_debug(f"_lnps_prioritize({conf['predicate_name']},{conf['arity']},{conf['weight']},{conf['modifier']}).")
        self._print_debug("lnps_config:", self.__lnps_config, log_level=LogLevel.FULL)

    def __check_variability(self):
        for conf in self.__lnps_config:
            if conf['weight'].type == SymbolType.Function:
                if conf['weight'].name == 'inf':
                    return False
        return True
    
    def __destroy(self, sol: str, conf: dict) -> list[Symbol]:
        projected_atoms = []
        for atom in sol.split(' '):
            symbol = parse_term(atom)
            if symbol.match(conf['predicate_name'], conf['arity']):
                projected_atoms.append(symbol)

        for a in projected_atoms:
            self._print_debug(f"atom projected by _lnps_project({conf['predicate_name']},{conf['arity']}):", a, log_level=LogLevel.MORE)
        self._print_debug(LINE, log_level=LogLevel.MORE)
                
        destroyed_atoms = []
        for i in range(len(conf['mask'])):
            filtered_atoms = self.__filter(projected_atoms, conf['arity'], conf['mask'][i], conf['pn'][i])
            if conf['pn'][i].arguments[0].number >= 0:
                destroyed_atoms.extend(filtered_atoms)
            elif conf['pn'][i].arguments[0].number < 0:
                destroyed_atoms.extend(list(set(projected_atoms)-set(filtered_atoms)))

        for a in destroyed_atoms:
            destroy_operators = ", ".join([f"_lnps_destroy({conf['predicate_name']},{conf['arity']},{conf['mask'][i]},{conf['pn'][i]})" for i in range(len(conf['mask']))])
            self._print_debug(f"atom destroyed by {destroy_operators}:", a, log_level=LogLevel.MORE)
        self._print_debug(LINE, log_level=LogLevel.MORE)
                
        prioritized_atoms = list(set(projected_atoms)-set(destroyed_atoms))

        for a in prioritized_atoms:
            self._print_debug(f"atom prioritized by _lnps_prioritize({conf['predicate_name']},{conf['arity']},{conf['weight']},{conf['modifier']}):", a, log_level=LogLevel.MORE)
        self._print_debug(LINE, log_level=LogLevel.MORE)
                
        return prioritized_atoms

    def __filter(self, atoms: Symbol, arity: int, mask: int, pn: Symbol) -> list[Symbol]:
        if mask == 0:
            return []
        else:
            bit_mask = list(format(mask,'0'+str(arity)+'b'))
            args_list = []
            for a in atoms:
                args = []
                for i in range(arity):
                    if int(bit_mask[i]):
                        args.append(a.arguments[i])
                args_list.append(args)
            unique_args_list = get_unique_list(args_list)

            for a in unique_args_list:
                self._print_debug(f"arguments filtered by {mask}({''.join(bit_mask)}):", f"({','.join([str(v) for v in a])})", log_level=LogLevel.FULL)
            self._print_debug(LINE, log_level=LogLevel.FULL)
            
            if pn.name == 'n':
                num = min(len(unique_args_list), abs(pn.arguments[0].number))
            elif pn.name == 'p':
                num = round(len(unique_args_list) * abs(pn.arguments[0].number) / 100)
            selected_args_list = random.sample(unique_args_list, num)

            for a in selected_args_list:
                self._print_debug(f"selected arguments ({pn}):", f"({','.join([str(v) for v in a])})", log_level=LogLevel.FULL)
            self._print_debug(LINE, log_level=LogLevel.FULL)

            filtered_atoms = []
            for a in atoms:
                args = []
                for i in range(arity):
                    if int(bit_mask[i]):
                        args.append(a.arguments[i])
                for selected_args in selected_args_list:
                    if args == selected_args:
                        filtered_atoms.append(a)

            for a in filtered_atoms:
                self._print_debug(f"filtered atom (mask:{mask}, pn:{pn}):", a, log_level=LogLevel.FULL)
            self._print_debug(LINE, log_level=LogLevel.FULL)
                        
            return filtered_atoms

    def __prioritize(self, targets: list[Symbol], conf: dict, step: int) -> list[Symbol]:
        heu_atoms = []
        for atom in targets:
            heu_atoms.append(Function('heuristic', [atom, conf['weight'], conf['modifier'], Number(step)]))
        return heu_atoms

    def __accept(self, cost_tmp: list[int], cost: list[int]) -> bool:
        threshold = cost[:-1]
        threshold.append(cost[-1] + abs(cost[-1]) * self._heulingo_config.acceptance_rate / 100)
        self._print_debug(LINE, log_level=LogLevel.FULL)
        self._print_debug("cost_tmp:", *cost_tmp, log_level=LogLevel.FULL)
        self._print_debug("cost:", *cost, log_level=LogLevel.FULL)
        self._print_debug("threshold:", *threshold, log_level=LogLevel.FULL)
        if cost_tmp < threshold:
            self._print_debug("acceptable", log_level=LogLevel.FULL)
            return True
        else:
            self._print_debug("unacceptable", log_level=LogLevel.FULL)
            return False
    
    def solve(self):
        self._ctl.ground([('base', [])])

        print_comment(LINE)
        print_comment("Finding an initial solution")
        print_comment(LINE)

        sol, cost = self._find()
        sol_best, cost_best = sol, cost

        if not self._finished:
            self.__calc_opt_bound(self.__iter_solver_config, cost)
            
            self._ctl.ground([('config', [])])
            self.__get_lnps_config(sol)

            self._print_debug(LINE)
            rules = ''
            self._print_debug("#program heuristic(t).")
            for c in self.__lnps_config:
                a = c['predicate_name'] + '(' + ','.join(['X'+str(i) for i in range(c['arity'])]) + ')'
                true_constraint = ':- not {0}, heuristic({0},inf,true,t).'.format(a)
                false_constraint = ':- {0}, heuristic({0},inf,false,t).'.format(a)
                heu_statement = '#heuristic {0} : heuristic({0},W,M,t), W != inf. [W,M]'.format(a)
                self._print_debug(true_constraint)
                self._print_debug(false_constraint)
                self._print_debug(heu_statement)
                rules += true_constraint + false_constraint + heu_statement
            self._ctl.add('heuristic', ['t'], rules)

            variability = self.__check_variability()
                    
            prev_heu_atoms = []

        step = 0
        while not self._finished:
            step += 1
            heu_atoms = []

            print_comment(LINE)
            print_comment("Iteration:", step)
            print_comment(LINE)

            for conf in self.__lnps_config:
                undestroyed = self.__destroy(sol, conf)
                heu_atoms.extend(self.__prioritize(undestroyed, conf, step))

            for a in prev_heu_atoms:
                self._ctl.release_external(a)
                self._print_debug(f"release external {a}.")
            self._print_debug(LINE)
		
            statements = ''
            for a in heu_atoms:
                ext_statement = f'#external {a}.'
                self._print_debug(ext_statement)
                statements += ext_statement
            self._print_debug(LINE)
            self._ctl.add('external', ['t'], statements)
            self._ctl.ground([('external', [Number(step)])])
            self._ctl.ground([('heuristic', [Number(step)])])

            for a in heu_atoms:
                self._ctl.assign_external(a, True)
                self._print_debug(f"assign external {a} True.")
            self._print_debug(LINE)
            prev_heu_atoms = heu_atoms.copy()            

            self._print_debug("objective value of current incumbent solution:", *cost)
            self._print_debug("objective value of current best solution:", *cost_best)
            self._print_debug(LINE)
            
            sol_tmp, cost_tmp = self._find(self.__iter_solver_config, variability)
            
            if self.__accept(cost_tmp, cost):
                sol, cost = sol_tmp, cost_tmp
                self.__calc_opt_bound(self.__iter_solver_config, cost)
            if cost_tmp < cost_best:
                sol_best, cost_best = sol_tmp, cost_tmp

            self.__increase_solve_limit(self.__iter_solver_config)

        print_comment(LINE)
        print_comment("Result")
        print_comment(LINE)
        print_variable("Answer:", sol_best)
        print_solution(self._result)
        print_comment("Iterations:", step)
        print_comment("Optimum:", self._optimum)
        print_answer("Optimization:", *cost_best)
        print_comment(LINE)
