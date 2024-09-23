#!/bin/bash

# Parse arguments
TOTAL_STEP=6
AIRCRAFT=20
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:2}

    case $par in
        a) AIRCRAFT=$value;;
        t) TOTAL_STEP=$value;;
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done

root=$(pwd)
python $root/instance/gen_init.py $AIRCRAFT
instance="$root/instance/init.lp $root/instance/mer_lmp.lp $root/instance/network.lp $root/instance/rq.lp"
encoding="$root/opt_lnps.lp $root/compute_path.lp"
lnsConfig="$root/lnps_config.lp"

# python3 $root/lnps-solver/heulingo.py --heulingo-configuration=tsp $encoding $instance $lnsConfig -c n=20 -c total_steps=$TOTAL_STEP

clingo $encoding $instance -c total_steps=$TOTAL_STEP