import sys
import random
import os
# if len(sys.argv) == 2:
#     cust = int(sys.argv[1])
#     random_flag = int(sys.argv[2])
#     out_filename = 'rq.lp'

cust = int(sys.argv[1])
random_flag = 0
out_filename = 'rq.lp'



vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
request_p = [[j, i] for i in range(0, 7) for j in range(0, 7) if i != j]
matrix_out = [[random.randint(0, cust) if i!=j else 0 for j in range(0, len(vPort_id))] for i in range(0, len(vPort_id))]

request_p = [[vPort_id[m[0]], vPort_id[m[1]]]for m in request_p]
request_print = []
total_edge = len(vPort_id) * (len(vPort_id)-1)
for i in range(0, total_edge):
    no = i
    origin = request_p[i][0] 
    destination = request_p[i][1]
    if random_flag == 0:
        request_print += [[no, (origin, destination), cust]]
    else:
        cust_rand = matrix_out[vPort_id.index(origin)][vPort_id.index(destination)]
        request_print += [[no, (origin, destination), cust_rand]]

f = open(f'instances/rq.lp',"w+")
f.write(f'%{matrix_out}')
f.write('%request(ID, (edge), number of request passenger)\n')
for i in request_print:
    f.write('request' + str(tuple(i)).replace("'","") + '.\n')

f.close()

    