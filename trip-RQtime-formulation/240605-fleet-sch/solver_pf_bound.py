# Solver for optimality gap finding with cost reduction
import argparse
import sys
import tempfile
import time
import copy
import subprocess
import re
import os
import pdb
#instance and program
input = 'rq.lp'
network = 'network.lp'
init = 'init.lp'
program = '8.0.1-planning.lp'
TIMESTEP = 6
TIME = 60
THREAD=1
TIME_LIMIT = 1000
parser = argparse.ArgumentParser(description='Process input and optional files.')
# parser.add_argument('nb_drone', type=str, help='number of drones')
# parser.add_argument('nb_request', type=str, help='number of requests')
# parser.add_argument('run_for_request', type=str, help='run for request')
# parser.add_argument('run_for_drone', type=str, help='run for drone')

args = parser.parse_args()

#pattern
pattern_opt = r"Optimization\s*:\s*(\d+)"
pattern_model = r"Models\s*:\s*(\d+)"
# #first run
# command = f"clingo path_edge_weight.lp init.lp opt.lp network.lp rq.lp mer_lmp.lp -q0 -t{THREAD} --time-limit={TIME} --outf=0 -c timestep={TIMESTEP}"
# output_from_terminal = subprocess.run(command, capture_output=True, text=True, shell=True)

# # Search for the pattern in the output text
# match = re.search(pattern_opt, output_from_terminal.stdout)

# # Extract and print the captured value if a match is found
# if match:
#     v = abs(int(match.group(1)))
#     print(f"Optimization value: {v}")

# max possible revenue 
v = 45

start = time.time()
end = time.time()
min = 0
max = v
c = round(abs(max/2))

#loop
while (end - start < TIME_LIMIT and min < max*0.9):
    print(f'min is {min}, max is {max}, goal is {c}')
    
    command = f"clingo path_edge_weight.lp init.lp opt_bound.lp network.lp rq_optimal_solution.lp mer_lmp.lp -q0 -t{THREAD} --outf=0 -c timestep={TIMESTEP} -c bound={c}"
    #output from terminal
    output_from_terminal = subprocess.run(command, capture_output=True, text=True, shell=True)
    #find if output from terminal have Model
    match = re.search(pattern_model, output_from_terminal.stdout)
    # print(f'output from terminal is: {output_from_terminal.stdout}')
    if match:
        if int(match.group(1))==1:
            min = c
            c = round(abs((min+max)/2))
            print('satisfied')
        else:
            max = c
            c = round(abs((min+max)/2))
            print('unsatisfied')
        # print(f'number of model is: {match.group(1)}')
    
    print('-------------------')
    
        