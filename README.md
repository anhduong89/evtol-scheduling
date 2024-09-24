## Note:
- v2: not using #heuristic, v2 is main.

- v3: using #heuristic, is in developed.
    - v3.2 is ready
## Rule explanation:
1. Each trajectory start by assigned to the longest trip.
2. For each segment `X>0` of agent `D`, choose the longest trip that has customer `cust(D,E,X)`.
3. For an agent `D`, at each segment `X`, every edge `(V,V')` generates atom `cust(D,E,X)` or `no_cust(D,E,X)`.
4. Atom `no_cust(D,E,X)` only appear with last trip `as_w(D',E,Y,10)`.
5. We use clingo-dl to specify the time of `no_cust` and `cust`.
6. For example, we are able to generate `cust(D,E,0), cust(D,E,1), no_cust(D,E,2)`.
## Content of encoding v3.2
1. cp_path_3.lp: generate the trajectory for all agents, also generate the flag `cust` or `no_cust` for each segment of agents on an edge. This is constrained by the last trip that has weight denoted as_w(D,E,X,10) (Each edge can have 3 trips with weight 4, 8, 10). 
2. time_schedule_3.lp: assignment of start time and arrival time to each trip. Assignment of weight.  
## Usage
To run with encoding v3.2: 
1. go to the folder 
`cd trip-noRQtime-formulation/v3-greedy-based/encoding_v3.2`

2. Run the solver
`bash solver_NY.sh`
