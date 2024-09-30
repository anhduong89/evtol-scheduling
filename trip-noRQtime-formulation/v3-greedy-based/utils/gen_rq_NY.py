import sys
import random
import os
weight = int(sys.argv[1])

# Get the absolute path of the current Python file
current_file_path = os.path.abspath(__file__)

# Get the directory name of the current Python file
root = os.path.dirname(os.path.dirname(current_file_path))

vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
request_p = [[j, i] for i in range(0, 7) for j in range(0, 7) if i != j]

request_p = [[vPort_id[m[0]], vPort_id[m[1]]]for m in request_p]
request_print = []
total_edge = len(vPort_id) * (len(vPort_id)-1)
for i in range(0, total_edge):
    no = i
    origin = request_p[i][0] 
    destination = request_p[i][1]
    request_print += [[no, (origin, destination), weight]]

f = open(f'{root}/rq.lp',"w+")
f.write('%request(ID, (edge), number of request passenger)\n')
for i in request_print:
    f.write('request' + str(tuple(i)).replace("'","") + '.\n')

f.close()