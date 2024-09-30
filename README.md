## Note:
- v3: using #heuristic, no #optimize
    - v3.3 is ready
## Rule explanation(depricate):
1. Each trajectory start by assigned to the longest trip.
2. For each segment `X>0` of agent `D`, choose the longest trip that has customer `cust(D,E,X)`.
3. For an agent `D`, at each segment `X`, every edge `(V,V')` generates atom `cust(D,E,X)` or `no_cust(D,E,X)`.
4. Atom `no_cust(D,E,X)` only appear with last trip `as_w(D',E,Y,10)`.
5. We use clingo-dl to specify the time of `no_cust` and `cust`.
6. For example, we are able to generate `cust(D,E,0), cust(D,E,1), no_cust(D,E,2)`.
## Content of encoding v3.3
1. path.lp
2. no_charge_opt.lp
3. charge_opt.lp (in developed)
## Usage
To run with encoding v3.3: 
1. go to the folder 
`cd trip-noRQtime-formulation/v3-greedy-based`
2. Modify the input within `solver_NY.sh`. For example: AGENT=10 is number of agents.
3. Run the solver
`bash solver_NY.sh`
