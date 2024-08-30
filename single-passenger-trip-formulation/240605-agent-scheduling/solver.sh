
#!/bin/bash
# Parse arguments
DEMAND=20
SEGMENT=27
TIME=1800
AGENT=10
ALLAS=0
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:1}

    case $par in
        a) AGENT=$value;;
        d) DEMAND=$value;;
        t) TIME=$value;;
        s) SEGMENT=$value;;
        q) ALLAS=$value;;
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done


DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "


# DEMAND=${1:-$DEFAULT_DEMAND}
# AGENT=${:-$DEFAULT_AGENT}
# TIME_LIMIT=${3:-$DEFAULT_TIME}
python gen_init.py $AGENT
# python gen_rq.py $DEMAND
echo "Segment Assignment; Number of Agents=$AGENT ;Max Running Time=$TIME; Segment=$SEGMENT " 
if [ $ALLAS -eq 1 ]; then
    clingo-dl path_greedy_rq.lp time_schedule.lp opt.lp init.lp network.lp rq.lp mer_lmp.lp -q0 --time-limit=$TIME -c max_segment=$SEGMENT
else
    clingo-dl path_greedy_rq.lp time_schedule.lp opt.lp init.lp network.lp rq.lp mer_lmp.lp -q1 --time-limit=$TIME -c max_segment=$SEGMENT
fi

# -n0: print all the answer set
# -q1: print last answer set