import sys
import random
import os
weight = int(sys.argv[1])

vPort_id = ["lbm", "mgt", "tci", "mbf", "crj"]
request_p = [[j, i] for i in range(0, 5) for j in range(0, 5) if i != j]

request_p = [[vPort_id[m[0]], vPort_id[m[1]]]for m in request_p]
request_print = []

for i in range(0, 20):
    no = i
    origin = request_p[i][0] 
    destination = request_p[i][1]
    request_print += [[no, (origin, destination), weight]]

f = open('rq.lp',"w+")
f.write('%request(ID, (edge), number of request passenger)\n')
for i in request_print:
    f.write('request' + str(tuple(i)).replace("'","") + '.\n')

f.close()