#const max_seg=6.  

#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

at(D, L, 0)      :- init_loc(D, L). 
remain_cust(E, R, 0) :- request(_,E, R). 

goal(E, 27) :- edge(E). 

step(1..max_seg).

% #program step(t).

% step(S). 

% -- assign edge to each segment
1 {as(D,(L,L'),0..C,S):edge((L,L'))} 1 :- agent(D), capacity(D,C), step(S), at(D, L, S-1).

% :- step(S), remain_cust(E, R, S-1), R > 0, as(D,E,0,S).

remain_cust(E,N-P,S) :- remain_cust(E,N,S-1), P=#sum{W,D: as(D,E,W,S)}, step(S).

at(D, L', S) :- step(S), at(D, L, S-1), as(D,(L,L'),_,S).

% total delivered customers can not excess demand
:- (R+1) #sum{W,D:as(D,E,W,S)}, edge(E), remain_cust(E,R,S-1), step(S).

% stationed aircraft does not carry customers
:- as(_,(V,V),W,t), W!=0.

% Prioritize scheduling higher-revenue flight
#heuristic as(_,E,W,S) : remain_cust(E,R,S-1). [W*R,true]

% #program check(S).

false(E, X, Y):- edge(E), goal(E, X), remain_cust(E, Y, max_seg), Y  > X.

#show step/1. 
#show as/4. 
#show remain_cust/3. 
#show at/3. 
#show false/3. 