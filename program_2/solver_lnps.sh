#!/bin/bash

# Parse arguments
TOTAL_STEP=6
AIRCRAFT=10
LNPS=True
for arg in "$@"
do
    par=${arg:0:1}
    value=${arg:2}

    case $par in
        a) AIRCRAFT=$value;;
        t) TOTAL_STEP=$value;;
        l) LNPS=$value;;
        *) echo "Unknown argument $key. Exiting."; exit 1;;
    esac
done

root=$(pwd)
# python $root/instanc`e/gen_init.py $AIRCRAFT
instance="$root/instance/init.lp $root/instance/mer_lmp.lp $root/instance/network.lp $root/instance/rq.lp"
encoding="$root/opt_lnps.lp $root/compute_path.lp"
lnsConfig="$root/lnps_config.lp"

if [ "$LNPS" = "True" ]; then
    timeout 1800 python3 $root/lnps-solver/heulingo.py --heulingo-configuration=tsp $encoding $instance $lnsConfig -c n=10 -c total_steps=$TOTAL_STEP
else
    clingo $encoding $instance --time-limit=1800 -c total_steps=$TOTAL_STEP
fi