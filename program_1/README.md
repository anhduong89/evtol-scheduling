# REQUIREMENT
1. python 3.9 (only python3.9 work with clingodl API)
2. numpy
3. pandas
4. clingo & clingo-dl
# FOLDERS & FILES

# USAGE

Stage 1: (assign flights and customers):

`clingo encoding/s.lp instances/network_NY_0.lp -c start_seg=1 -c max_seg=13 --heuristic=Domain -t4 --outf=1`

Stage 2 (switching):

`clingo encoding/swap0.1.lp instances/network_NY_0.lp test_1.lp -c horizon=180 --out-ifs=\\n --out-atomf=%s.`
<br> test_1 is the flight path. Flight path is the asp facts of predicate 'as'.

Stage 3 (assign time with clingo-dl):

`clingo-dl encoding/time0.lp test_2.lp --out-ifs=\\n --out-atomf=%s.`
<br> test_2 is the flight path

Run all stages:
`python solverClingoAPI.py --n_agents=34 --n_rq=30 --horizon=180 --max_segment=13`


