#!/bin/bash
# run only maximize the revenue; agent start with 0 battery and operation time is equivalent to sum of charge time + fly time.

# Parse arguments
DEMAND=20
SEGMENT=10
TIME=1800
AGENT=10
THREAD=1
TIMESTEP=11
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:1}

    case $par in
        a) AGENT=$value;;
        d) DEMAND=$value;;
        t) TIME=$value;;
        s) TIMESTEP=$value;;
        p) THREAD=$value;;

        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done


DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "

root=$(pwd)  

instance="$root/instance/init.lp $root/instance/mer_lmp.lp $root/instance/network.lp $root/instance/rq_small.lp"
encoding="$root/encoding/opt_max_rev.lp $root/encoding/compute_path.lp"

# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=${:-$DEFAULT_AGENT}
# TIME_LIMIT=${3:-$DEFAULT_TIME}
# python gen_init.py $AGENT
# python gen_rq.py $DEMAND
# clingo-dl path_edge_weight.lp time_schedule.lp opt.lp init.lp network.lp rq.lp mer_lmp.lp -n0 --time-limit=$TIME
echo "Weighted Edge-time Assignment; Number of Agents=$AGENT ;Max Running Time=$TIME; timestep=$TIMESTEP"
clingo $instance $encoding -q0 -t$THREAD --time-limit=$TIME --outf=0 -c timestep=$TIMESTEP --out-atomf=%s. #| tr ' ' '\n' 
# clingo path_edge_weight.lp init.lp opt.lp network.lp rq.lp mer_lmp.lp -n0 -t$THREAD --time-limit=$TIME
# clingo-dl ${DEFAULT_CONF}