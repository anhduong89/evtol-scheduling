"""
Heulingo
@author Irumi Sugimori
Time-stamp: <2024-04-06 17:01:17 irumisugimori>
"""

import sys
import re
from typing import Any, Callable, Sequence

from clingo.application import clingo_main, Application, ApplicationOptions
from clingo.control import Control

from solver import HeulingoConfig, LogLevel, print_warning
from solver_lnps import SolverLNPS


BREAK = """
      """


def parse_iter_configuration(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        ctl = Control()
        try:
            ctl.configuration.configuration = sval
        except RuntimeError:
            return False
        config.iter_configuration = sval
        return True
    return parse

def parse_iter_opt_strategy(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        ctl = Control()
        try:
            ctl.configuration.solver.opt_strategy = sval
        except RuntimeError:
            return False
        config.iter_opt_strategy = sval
        return True
    return parse

def parse_iter_opt_heuristic(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        ctl = Control()
        try:
            ctl.configuration.solver.opt_heuristic = sval
        except RuntimeError:
            return False
        config.iter_opt_heuristic = sval
        return True
    return parse

def parse_iter_heuristic(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        ctl = Control()
        try:
            ctl.configuration.solver.heuristic = sval
        except RuntimeError:
            return False
        config.iter_heuristic = sval
        return True
    return parse

def parse_iter_opt_mode(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        values = sval.split(",")
        if values[0] not in ("opt", "enum", "optN", "ignore"):
            return False
        config.iter_opt_mode['mode'] = values[0]        
        if len(values) == 1:
            config.iter_opt_mode['nf'] = None
            config.iter_opt_mode['modifier'] = None
        elif len(values) == 2:
            try:
                float(values[1])
            except ValueError:
                return False
            config.iter_opt_mode['nf'] = float(values[1])
            config.iter_opt_mode['modifier'] = "dynamic"
        elif len(values) >= 3:
            if values[-1] == "static":
                try:
                    for v in values[1:-1]:
                        int(v)
                except ValueError:
                    return False
                config.iter_opt_mode['nf'] = ",".join(values[1:-1])
                config.iter_opt_mode['modifier'] = "static"
            elif values[-1] == "dynamic":
                if len(values) >= 4:
                    return False
                try:
                    float(values[1])
                except ValueError:
                    return False
                config.iter_opt_mode['nf'] = float(values[1])
                config.iter_opt_mode['modifier'] = "dynamic"
            else:
                return False
        else:
            return False
        return True
    return parse

def parse_iter_solve_limit(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        ctl = Control()
        try:
            ctl.configuration.solve.solve_limit = sval
        except RuntimeError:
            return False
        config.iter_solve_limit = sval
        return True
    return parse

def parse_float(config: Any, attr: str) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        try:
            float(sval)
        except ValueError:
            return False
        setattr(config, attr, float(sval))
        return True
    return parse

def parse_random_seed(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        if sval == "no":
            config.random_seed = None
        else:
            try:
                int(sval)
            except ValueError:
                return False
            config.random_seed = int(sval)
        return True
    return parse

def parse_heulingo_configuration(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        if sval not in ("teaspoon", "tsp", "sgp", "spg", "wsc", "wsc,large", "wsc,medium", "sd"):
            return False
        if sval == "wsc":
            config.heulingo_configuration = sval + ",medium"
        else:
            config.heulingo_configuration = sval
        return True
    return parse

def parse_log_level(config: Any) -> Callable[[str], bool]:
    def parse(sval: str) -> bool:
        try:
            int(sval)
        except ValueError:
            return False
        num = int(sval)
        if num in [LogLevel.NONE, LogLevel.BASIC, LogLevel.MORE, LogLevel.FULL]:
            config.log_level = num
        else:
            return False
        return True
    return parse


class HeulingoApp(Application):

    def __init__(self, argv):
        self.program_name = "heulingo"
        self.version = "0.2"
        self.__argv = argv
        self.__config = HeulingoConfig()
        
    def register_options(self, options: ApplicationOptions):
        group = "Heulingo Options"
        
        options.add(
            group, "iter-configuration",
            "Set default configuration in iterations",
            parse_iter_configuration(self.__config),
            argument="<arg>")

        options.add(
            group, "iter-opt-strategy",
            "Configure optimization strategy in iterations",
            parse_iter_opt_strategy(self.__config),
            argument="<arg>")

        options.add(
            group, "iter-opt-heuristic",
            "Use opt in iterations. in <list {{sign|model}}> heuristics",
            parse_iter_opt_heuristic(self.__config),
            argument="<list>")

        options.add_flag(
            group, "iter-restart-on-model",
            "Restart after each model in iterations",
            self.__config.iter_restart_on_model)

        options.add(
            group, "iter-heuristic,@2",
            f"Configure decision heuristic in iterations [{self.__config.iter_heuristic}]",
            parse_iter_heuristic(self.__config),
            argument="<heu>")
        
        options.add(
            group, "iter-opt-mode",
            (f"Configure optimization algorithm in iterations{BREAK}"
             f"<arg>: <mode>[,<bound>]{BREAK}"
             f"  <mode> : {{opt|enum|optN|ignore}}{BREAK}"
             f"    opt   : Find optimal model{BREAK}"
             f"    enum  : Find models with costs <= initial bound set based on <bound>{BREAK}"
             f"    optN  : Find optimum, then enumerate optimal models{BREAK}"
             f"    ignore: Ignore optimize statements{BREAK}"             
             f"  <bound>: {{<n>...,static|<f>[,dynamic]}}{BREAK}"
             f"    <n>...,static: In every iteration, set <n>... as initial bound for objective function(s){BREAK}"
             f"    <f>[,dynamic]: Set initial bound for objective function(s) in each iteration such that{BREAK}"
             f"                   solutions whose objective value is at least <f>%% worse than{BREAK}"
             f"                   current incumbent solution are not obtained"),
            parse_iter_opt_mode(self.__config),
            argument="<arg>")

        options.add(
            group, "iter-solve-limit",
            "Stop search after <n> conflicts or <m> restarts in iterations",
            parse_iter_solve_limit(self.__config),
            argument="<n>[,<m>]")

        options.add(
            group, "solve-limit-increase-rate",
            f"Increase the number of conflicts and restarts, condition for stopping search, by <f>%% in each iteration [{self.__config.solve_limit_increase_rate}]",
            parse_float(self.__config, 'solve_limit_increase_rate'),
            argument="<f>")

        options.add(
            group, "acceptance-rate",
            f"Do not accept solution whose objective value is at least <f>%% worse than current incumbent solution in each iteration [{self.__config.acceptance_rate}]",
            parse_float(self.__config, 'acceptance_rate'),
            argument="<f>")

        options.add(
            group, "random-seed",
            f"Set <n> as seed of random number generator used in Python's random module [{self.__config.random_seed}]",
            parse_random_seed(self.__config),
            argument="<n>|no")

        options.add(
            group, "heulingo-configuration",
            (f"Set default configuration{BREAK}"
             f"<arg>: {{teaspoon|tsp|sgp|spg|wsc[,<scale>]|sd}}{BREAK}"
             f"  teaspoon: Use defaults geared towards CB-CTT problems{BREAK}"
             f"  tsp     : Use defaults geared towards traveling salesperson problem{BREAK}"
             f"  sgp     : Use defaults geared towards social golfer problem{BREAK}"
             f"  spg     : Use defaults geared towards sudoku puzzle generation{BREAK}"
             f"  wsc     : Use defaults geared towards weighted strategic companies{BREAK}"
             f"    <scale>: Use defaults geared towards {{medium|large}} instances [medium]{BREAK}"
             f"  sd      : Use defaults geared towards shift design{BREAK}"
             f"Default configurations:{BREAK}"
             f"[teaspoon]:{BREAK}"
             f" --configuration={HeulingoConfig.heulingo_configuration_values['teaspoon']['configuration']}"
             f" --opt-strategy={HeulingoConfig.heulingo_configuration_values['teaspoon']['opt-strategy']}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['teaspoon']['solve-limit']}{BREAK}"
             f" --iter-configuration={HeulingoConfig.heulingo_configuration_values['teaspoon']['iter-configuration']}"
             f" --iter-opt-strategy={HeulingoConfig.heulingo_configuration_values['teaspoon']['iter-opt-strategy']}"
             f" --iter-opt-heuristic={HeulingoConfig.heulingo_configuration_values['teaspoon']['iter-opt-heuristic']}{BREAK}"
             f" --iter-restart-on-model"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['teaspoon']['iter-solve-limit']}{BREAK}"
             f"[tsp]:{BREAK}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['tsp']['solve-limit']}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['tsp']['iter-solve-limit']}{BREAK}"
             f"[sgp]:{BREAK}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['sgp']['solve-limit']}"
             f" --iter-opt-mode={HeulingoConfig.heulingo_configuration_values['sgp']['iter-opt-mode']}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['sgp']['iter-solve-limit']}{BREAK}"
             f"[spg]:{BREAK}"
             f" --configuration={HeulingoConfig.heulingo_configuration_values['spg']['configuration']}"
             f" -t{HeulingoConfig.heulingo_configuration_values['spg']['parallel-mode']}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['spg']['solve-limit']}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['spg']['iter-solve-limit']}{BREAK}"
             f"[wsc,medium]:{BREAK}"
             f" --opt-strategy={HeulingoConfig.heulingo_configuration_values['wsc,medium']['opt-strategy']}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['wsc,medium']['solve-limit']}"
             f" --iter-opt-strategy={HeulingoConfig.heulingo_configuration_values['wsc,medium']['iter-opt-strategy']}{BREAK}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['wsc,medium']['iter-solve-limit']}{BREAK}"
             f"[wsc,large]:{BREAK}"
             f" --opt-strategy={HeulingoConfig.heulingo_configuration_values['wsc,large']['opt-strategy']}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['wsc,large']['solve-limit']}"
             f" --iter-opt-strategy={HeulingoConfig.heulingo_configuration_values['wsc,large']['iter-opt-strategy']}{BREAK}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['wsc,large']['iter-solve-limit']}{BREAK}"
             f"[sd]:{BREAK}"
             f" --configuration={HeulingoConfig.heulingo_configuration_values['sd']['configuration']}"
             f" --opt-strategy={HeulingoConfig.heulingo_configuration_values['sd']['opt-strategy']}"
             f" --solve-limit={HeulingoConfig.heulingo_configuration_values['sd']['solve-limit']}{BREAK}"
             f" --iter-opt-mode={HeulingoConfig.heulingo_configuration_values['sd']['iter-opt-mode']}"
             f" --iter-solve-limit={HeulingoConfig.heulingo_configuration_values['sd']['iter-solve-limit']}"),
            parse_heulingo_configuration(self.__config),
            argument="<arg>")        

        options.add(
            group, "log-level",
            f"Specify verbosity of output {{{LogLevel.NONE}=none|{LogLevel.BASIC}=basic|{LogLevel.MORE}=more|{LogLevel.FULL}=full}} [{self.__config.log_level}]",
            parse_log_level(self.__config),
            argument="<n>")

    def validate_options(self) -> bool:
        if self.__config.iter_opt_mode['modifier'] == "dynamic":
            if self.__config.iter_opt_mode['nf'] <= 0:
                print_warning("heulingo may finish without proving optimality because of 0 or less percent of iter-opt-mode")
            if self.__config.iter_opt_mode['nf'] < self.__config.acceptance_rate:
                print_warning("Rate of iter-opt-mode is less than rate of acceptance-rate")
        return True
                              
    def main(self, ctl: Control, files: Sequence[str]):
        for arg in self.__argv:
            if re.match(r"--(no-)?iter-r", arg):
                self.__config.has_iter_restart_on_model = True

        if not files:
            files = ["-"]
        for file_ in files:
            ctl.load(file_)

        solver = SolverLNPS(ctl, self.__config)
        solver.solve()

        
if __name__ == "__main__":
    argv = sys.argv[1:].copy()
    argv_length = len(argv)
    new_argv = argv.copy()
    
    has_configuration = False
    has_opt_strategy = False
    has_parallel_mode = False
    has_solve_limit = False
    has_iter_opt_mode= False
    for arg in argv:
        if re.match(r"--conf", arg):
            has_configuration = True
        if re.match(r"--opt-st", arg):
            has_opt_strategy = True
        if re.match(r"-t", arg) or re.match(r"--para", arg):
            has_parallel_mode = True
        if re.match(r"--solve-limit", arg) and not re.match(r"--solve-limit-", arg):
            has_solve_limit = True
        if re.match(r"--iter-opt-m", arg):
            has_iter_opt_mode = True
                
    has_heulingo_configration = False
    for i in range(argv_length):
        if re.match(r"--heul", argv[i]):
            has_heulingo_configration = True
            heulingo_configuration = None
            if re.fullmatch(r".*=teaspoon", argv[i]) or re.fullmatch(r"teaspoon", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "teaspoon"
            elif re.fullmatch(r".*=tsp", argv[i]) or re.fullmatch(r"tsp", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "tsp"
            elif re.fullmatch(r".*=sgp", argv[i]) or re.fullmatch(r"sgp", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "sgp"
            elif re.fullmatch(r".*=spg", argv[i]) or re.fullmatch(r"spg", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "spg"
            elif (re.fullmatch(r".*=wsc,medium", argv[i]) or re.fullmatch(r"wsc,medium", argv[min(i+1, argv_length-1)]) or
                  re.fullmatch(r".*=wsc", argv[i]) or re.fullmatch(r"wsc", argv[min(i+1, argv_length-1)])):
                heulingo_configuration = "wsc,medium"
            elif re.fullmatch(r".*=wsc,large", argv[i]) or re.fullmatch(r"wsc,large", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "wsc,large"
            elif re.fullmatch(r".*=sd", argv[i]) or re.fullmatch(r"sd", argv[min(i+1, argv_length-1)]):
                heulingo_configuration = "sd"

            if heulingo_configuration is not None:
                configuration = HeulingoConfig.heulingo_configuration_values[heulingo_configuration]['configuration']
                opt_strategy = HeulingoConfig.heulingo_configuration_values[heulingo_configuration]['opt-strategy']
                parallel_mode = HeulingoConfig.heulingo_configuration_values[heulingo_configuration]['parallel-mode']
                solve_limit = HeulingoConfig.heulingo_configuration_values[heulingo_configuration]['solve-limit']
                iter_opt_mode = HeulingoConfig.heulingo_configuration_values[heulingo_configuration]['iter-opt-mode']
                if not has_configuration and configuration is not None:
                    new_argv.append(f"--configuration={configuration}")
                if not has_opt_strategy and opt_strategy is not None:
                    new_argv.append(f"--opt-strategy={opt_strategy}")
                if not has_parallel_mode and parallel_mode is not None:
                    new_argv.append(f"-t{parallel_mode}")
                if not has_solve_limit and solve_limit is not None:
                    new_argv.append(f"--solve-limit={solve_limit}")
                if not has_iter_opt_mode and iter_opt_mode is not None:
                    new_argv.append(f"--iter-opt-mode={iter_opt_mode}")
            break
        
    if not has_heulingo_configration:
        has_solve_limit = False
        for arg in argv:
            if re.match(r"--so", arg):
                has_solve_limit = True
        if not has_solve_limit:
            new_argv.append("--solve-limit=2500000,5000")

    clingo_main(HeulingoApp(argv), new_argv)
