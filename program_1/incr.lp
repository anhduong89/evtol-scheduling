#include <incmode>.

#program base.

#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

at(D, L, 0)      :- init_loc(D, L). 
remain_cust(E, R, 0) :- request(_,E, R). 

goal(E, 0) :- edge(E). 

#program step(t).

time(t). 

% -- assign edge to each segment
1 {as(D,(L,L'),0..C,t):edge((L,L'))} 1 :- 
    agent(D), capacity(D,C), time(t), at(D, L, t-1).

:- time(t), remain_cust(E, R, t), R > 0, #count{W,D: as(D,E,W,t),  W > 0} = 0.

remain_cust(E,N-P,t) :- remain_cust(E,N,t-1), P=#sum{W,D: as(D,E,W,t)}, time(t).

at(D, L', t) :- time(t), at(D, L, t-1), as(D,(L,L'),_,t-1), t > 1.

% total delivered customers can not excess demand
:- (R+1) #sum{W,D:as(D,E,W,t)}, edge(E), remain_cust(E,R,t-1), time(t).

% stationed aircraft does not carry customers
:- as(_,(V,V),W,t), W!=0.

% Prioritize scheduling higher-revenue flight
% #heuristic as(_,_,W,_). [W,true]

#program check(t).

:- query(t), edge(E), goal(E, X), remain_cust(E, Y, t), Y  > X.

% #show seq/1. 
% #show as/4. 

#show remain_cust/3. 