
#!/bin/bash
# Parse arguments
DEMAND=20
SEGMENT=10
TIME=1800
AGENT=10
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:1}

    case $par in
        a) AGENT=$value;;
        d) DEMAND=$value;;
        t) TIME=$value;;
        s) SEGMENT=$value;;
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done


DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "


# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=$2
# TIME_LIMIT=${3:-$DEFAULT_TIME}
python gen_init.py $AGENT
python gen_rq.py $DEMAND
clingo-dl cp_path.lp time_schedule.lp opt.lp init.lp network.lp rq.lp mer_lmp.lp -n0 --time-limit=$TIME
# clingo-dl ${DEFAULT_CONF}