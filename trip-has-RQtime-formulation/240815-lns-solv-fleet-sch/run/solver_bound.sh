
#!/bin/bash
# Parse arguments
DEMAND=20
SEGMENT=10
TIME=60
AGENT=30
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


# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=${:-$DEFAULT_AGENT}
# TIME_LIMIT=${3:-$DEFAULT_TIME}
python gen_init.py $AGENT
# python gen_rq.py $DEMAND
# clingo-dl path_edge_weight.lp time_schedule.lp opt.lp init.lp network.lp rq.lp mer_lmp.lp -n0 --time-limit=$TIME
echo "Weighted Edge-time Assignment; Number of Agents=$AGENT; Max Running Time=$TIME; timestep=$TIMESTEP"

# first run
# clingo path_edge_weight.lp init.lp opt.lp network.lp rq.lp mer_lmp.lp -q1 -t$THREAD --time-limit=$TIME --outf=0 -c timestep=$TIMESTEP | grep "Optimization:"

optimization_value=$(clingo path_edge_weight.lp init.lp opt.lp network.lp rq.lp mer_lmp.lp -q1 -t$THREAD --time-limit=$TIME --outf=0 -c timestep=$TIMESTEP | grep "Optimization:" | awk '{print $2}' | tr -d '-')

echo "Captured Optimization Value: $optimization_value"

for i in {1..10}; do
    clingo_output=$(clingo path_edge_weight.lp init.lp opt_bound.lp network.lp rq.lp mer_lmp.lp -q1 -t$THREAD --outf=0 -c timestep=$TIMESTEP -c bound=$optimization_value )

    echo "$clingo_output"

    optimization_value=$(echo "$clingo_output" | grep -o 'cost([0-9]*)' | awk -F'[()]' '{print $2}')
    echo "Run number $i ;Extracted Cost Values: $optimization_value"
done

# # Catch the whole line that contains 'Time:'
# time_line=$(echo "$clingo_output" | grep 'Time:')

# # Echo the caught line
# echo "Caught Line: $time_line"