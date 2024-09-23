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
clingo $instance $encoding -q0 --outf=0
# clingo path_edge_weight.lp init.lp opt.lp network.lp rq.lp mer_lmp.lp -n0 -t$THREAD --time-limit=$TIME
# clingo-dl ${DEFAULT_CONF}