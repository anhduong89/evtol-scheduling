% ------ optimization_2: penalize unserved customers.
#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

remain_cust(E,N-P) :- request(_,E,N), P=#sum{W,D,X: as(D,E,W,X)}, edge(E).

total_remain(W) :- W = #sum{P,E:remain_cust(E,P)}.
% #show remain_cust/2.
#show total_remain/1.
