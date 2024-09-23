
#!/bin/bash
# default setting
DEMAND=10
MAX_SEGMENT=6
TIME=600
AGENT=10
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
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done


DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "


# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=$2
# TIME_LIMIT=${3:-$DEFAULT_TIME}
python utils/gen_init_NY.py $AGENT
# % python gen_rq.py $DEMAND
if [ "$TEST" == "1" ]
then
    clingo-dl cp_path.lp opt.lp init.lp network_NY.lp rq.lp time_schedule.lp mer_lmp_mile.lp
else
    echo "MAX_SEG=${MAX_SEGMENT} TIME-LIMIT=${TIME} AGENT=${AGENT} NETWORK=NY NO COST"
    clingo-dl cp_path.lp opt.lp init.lp network_NY.lp rq.lp time_schedule.lp mer_lmp_mile.lp -c max_seg=$MAX_SEGMENT --time-limit=${TIME}
fi
# clingo-dl ${DEFAULT_CONF}