import sys
import argparse
import os
import subprocess
from utils.sort_answer_set import sort_answer_set
from utils.get_answer_set import get_answer_set
# Argument parsing
parser = argparse.ArgumentParser(description='Solver for eVTOL scheduling.')
parser.add_argument('--n_rq', type=int, default = 30, help='Number of requests')
parser.add_argument('--n_agents', type=int, default = 34, help='Number of agents')
parser.add_argument('--max_segment', type=int, default = 10, help='Maximum segment')
parser.add_argument('--horizon', type=int, default = 180, help='Horizon')
parser.add_argument('--time_limit', type=int, default = 30, help='Time limit')
args = parser.parse_args()

if args.n_agents is not None:
    gen_init = ["python"
                ,"instances/gen_init_random_NY.py"
                , str(args.n_agents)
                , str(1)
                , str(6)
                ]
    subprocess.run(gen_init)  # Run gen_init script
if args.n_rq is not None:
    gen_rq = ["python"
            , "instances/gen_rq_NY.py"
            , str(args.n_rq)
            ]
    subprocess.run(gen_rq)  # Run gen_rq script

clingo_schedule = ["clingo-dl"
                   , "schedule.lp"
                   , f"-c start_seg=0"
                   , f"-c max_seg={args.max_segment}"
                   , f"-c horizon={args.horizon}"
                   , f"--time-limit={args.time_limit}"
                   , "--heuristic=Domain"
                   , '-q1,1'
                   ]

process = subprocess.Popen(clingo_schedule, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

result_stdout = []
result_stderr = []

for stdout_line in iter(process.stdout.readline, ""):
    print(stdout_line, end='')
    result_stdout.append(stdout_line)
for stderr_line in iter(process.stderr.readline, ""):
    print(stderr_line, end='', file=sys.stderr)
    result_stderr.append(stderr_line)

process.stdout.close()
process.stderr.close()
process.wait()

result = subprocess.CompletedProcess(
    args=clingo_schedule,
    returncode=process.returncode,
    stdout=''.join(result_stdout),
    stderr=''.join(result_stderr)
)

# compute revenue
clingo_compute_revenue = ["clingo"
                          ,"compute_revenue.lp"
                          ,"results/result_answer_set.lp"]

# save output from scheduling
get_answer_set(result.stdout)
with open(fr'results/result.lp', 'w') as txt_file:
    txt_file.write(result.stdout)

# output from compute revenue
process = subprocess.Popen(clingo_compute_revenue, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
for stdout_line in iter(process.stdout.readline, ""):
    print(stdout_line, end='')
    result_stdout.append(stdout_line)
for stderr_line in iter(process.stderr.readline, ""):
    print(stderr_line, end='', file=sys.stderr)
    result_stderr.append(stderr_line)
total_revenue, total_em_cost, total_chg_cost, profit = sort_answer_set(result.stdout)


