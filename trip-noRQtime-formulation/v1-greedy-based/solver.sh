
#!/bin/bash
DEFAULT_CONF=" cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp --outf=0 -V0 -q1 --out-atomf=%s. | tr ' ' '\n'  "

python gen_init.py $1
clingo-dl cp_path.lp assign_charge.lp time_schedule.lp opt.lp init.lp base_data.lp -n0
# clingo-dl ${DEFAULT_CONF}