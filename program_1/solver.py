import sys
import argparse
import os
import subprocess
from sort_answer_set import sort_answer_set

# Argument parsing
parser = argparse.ArgumentParser(description='Solver for eVTOL scheduling.')
parser.add_argument('--n_agents', type=int, default = None, help='Number of agents')
parser.add_argument('--n_rq', type=int, default = None, help='Number of requests')
parser.add_argument('--max_segment', type=int, help='Maximum segment')
parser.add_argument('--horizon', type=int, default = 180, help='Horizon')
parser.add_argument('--time_limit', type=int, default = 300, help='Time limit')
args = parser.parse_args()

if args.n_agents is not None:
    gen_init = ["python"
                ,"instances/gen_init_NY.py"
                , str(args.n_agents)
                ]
if args.n_rq is not None:
    gen_rq = ["python"
            , "instances/gen_rq_NY.py"
            , str(args.n_rq)
            ]

clingo_schedule = ["clingo-dl"
                   , "schedule.lp"
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
# run_time = None
# optimization = None
# for line in result.stdout.splitlines():
#     if line.startswith("Time         :"):
#         run_time = float(line.split("Time         :")[1].strip().split('s')[0])
#     if line.startswith("Optimization :"):
#         optimization = int(line.split("Optimization :")[1].strip())

total_revenue, total_em_cost, total_chg_cost, profit = sort_answer_set(result.stdout)
# print("--- instances information ----\n")
# print(f'Max possible revenue: {args.agents}')
# print({'revenue':total_revenue, 'em_cost':total_em_cost, 'charge_cost': total_chg_cost,'profit': profit, 'TIME': run_time, 'optimization': optimization})

