# REQUIREMENT
1. python 3.x (I'm running python 3.10)
2. numpy
3. pandas
4. clingo & clingo-dl
# FOLDERS & FILES
1. instances/ : contain instances and instances generator
2. schedule.lp : answer set program for eVTOL schedule
3. solver.py : solve schedule.lp and compute auxiliary value
4. penalty.lp : answer set program for minimize number of empty flights
5. MER_LMP_Information.xlsx : constant value to compute charging cost and emission cost.
6. sort_answer_set.py : sort the flights by agent index and segment index for visualization, and compute revenue, profit, cost.
7. results/ : contain output from sort_answer_set.py (use to display answer set in sorted order)
# USAGE
There are two ways to run the schedule (produce the same result with same input values):

**(1)**:

specify number of agents, for example = 15:
`python instances/gen_rq_NY.py 15`

specify customers each edge, for example = 20:
`python instances/gen_init_NY.py 20`

solve:
`clingo-dl schedule.lp -c max_seg=11 -c horizon=180 --heuristic=Domain`


**(2)**:
`python solver.py --n_agents 15 --n_rq 20 --max_segment 10`

_Note:_

(1) directly pass program `schedule.lp` to clingo-dl. Instances specification are included in the program. 

(2) `solver.py` first solving the program `schedule.lp` with clingo-dl, then compute the revenue, profit for last answer set produced from solving `schedule.lp`.

To run the program `schedule.lp` with optimization directive `penalty.lp`:

**(1.1)**: 
`clingo-dl schedule.lp penalty.lp -c max_seg=11 -c horizon=180 --heuristic=Domain`
# PARAMETER
For method **(1)**:


| parameter | detail |
|-------------|---------|
|max_seg| number of segments for each agents|
|horizon| planning horizon|


For method **(2)**:
|parameter|detail|
|-------------|---------|
|--n_agents| number of agents|
|--n_rq| number of demand customers each edge|
|--max_segment| number of segments each agents can assign flights|
|--horizon| planning horizon (in minutes)|
|--time_limit | run time limit |