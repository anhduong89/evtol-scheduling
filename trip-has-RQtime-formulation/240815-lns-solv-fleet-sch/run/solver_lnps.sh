#!/bin/bash
root='/Users/duong/Work-Documents/NMSU/NASA_ULI_2022/my-code/drone-scheduling/lns-solving'

instance="$root/instance/init.lp $root/instance/mer_lmp.lp $root/instance/network.lp $root/instance/rq.lp"
encoding="$root/encoding/opt_lnps.lp $root/encoding/path.lp"
lnsConfig="$root/encoding/lnps_config.lp"

python3 $root/lnps-solver/heulingo.py --heulingo-configuration=tsp $encoding $instance $lnsConfig -c n=20