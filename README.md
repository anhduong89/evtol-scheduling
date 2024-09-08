# Requirement
1. Clingo-dl
2. Python 3.10
# Content
| File | Content |
|-------------|---------|
|instance/gen_init.py| initialization of aircraft number and location
|instance/init.lp | aircraft location 
|instance/mer_lmp.lp|LMP value (MER is not yet available)|
| instance/network.lp| vertiport network|
| instance/rq.lp| trip request
| compute_path.lp | compute fleet trajectories
| opt_lnps.lp | optimization specification
| time_schedule.lp | compute departure and arrival time of fleet
| solver_lnps.sh | run solver
| lnps_config.lp | LNPS config
# Predicate description
|File|Atom/Predicate|Description
|-------------|---------|----------------
|instance/init.lp | init_loc(0, mgt) | agent 0 start at vertiport 'mgt'
|instance/mer_lmp.lp|lmp(25, lbm, 10)| LMP value = 25 at vertiport 'lbm' at time '10'
| instance/network.lp| distance((lbm, mgt), 130) | edge (lbm, mgt) has distance 130
| instance/rq.lp| request(0,crj,mbf,5,0) | request ID=0 of 5 passengers, ask to travel on the edge (crj,mbf) at step 0
| compute_path.lp | node_w(V, W, T) | there is W/4 aircraft staying at vertiport V at step T
|  | edge_w(E, W, T) | there is W/4 aircraft departure at V at step T on edge E=(V, V').  
| opt_lnps.lp | revenue(N, R)| yield R revenue for request N
| time_schedule.lp | not used yet

# Hyperparameter
|Parameter|Value
|---------|----------
|a| total number of aircraft; default = 20
|t| total number of steps; default = 6 (3 hours)
# Usage

Solving with default parameters:

`./solver_lnps.sh`

Solving with config 40 aircraft and in 11 steps horizon:

`./solver_lnps.sh a=40 t=11`

notice: 
- for testing with small number of requests: replace *rq.lp* in **solver_lnps.sh** by *rq_small.lp*
