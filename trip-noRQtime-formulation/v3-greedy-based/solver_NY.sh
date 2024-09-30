
#!/bin/bash
# default setting
DEMAND=10
MAX_SEGMENT=12
AGENT=5
HORIZON=180
OPT_BOUND=0
RUNNING_TIME=1800
# Parse arguments
# to set the value; try "bash solver_NY.sh a=10 t=600 s=12"
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:2}

    case $par in
        a) AGENT=$value;;
        d) DEMAND=$value;;
        t) TIME=$value;;
        s) MAX_SEGMENT=$value;;
        b) OPT_BOUND=$value;;
        h) HORIZON=$value;;
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done

PARENT_DIR=$(realpath ../)
# DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "


# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=$2
# TIME_LIMIT=${3:-$DEFAULT_TIME}
python utils/gen_init_NY.py $AGENT
# python utils/gen_rq_NY.py $DEMAND
echo "MAX_SEG=${MAX_SEGMENT} /TIME-LIMIT=${TIME}/ NB_OF_AGENTS=${AGENT} / NETWORK=NY / NO COST /HEURISTIC"
clingo-dl  run_encodingv3.3_no_opt_charge.lp instance/instance_NY.lp -c max_seg=$MAX_SEGMENT -c horizon=${HORIZON} --heuristic=Domain --time-limit=${RUNNING_TIME}
# clingo-dl ${DEFAULT_CONF}