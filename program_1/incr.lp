#include <incmode>.

#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".


#program base.


at(D, L, 0)      :- init_loc(D, L). 

remain_cust(E, R, 0) :- request(_,E, R). 

remain_loc(L, R, 0)  :- vertiport(L), R = #sum{P, L' : request(_, (L, L'), P)}. 

goal(E, 0) :- edge(E), not edge_loop(E).

%hold(E, 0) :- edge(E), goal(E, X), not edge_loop(E), remain_cust(E, Y, 0), Y <= X. 

% hold(E, 0) :- goal(E, X), remain_cust(E, Y, 0), Y <= X. 


#program step(t).

time(t). 

% hold(E, t) :- edge(E), goal(E, X), remain_cust(E, Y, 1), Y < X.

% hold(E, t) :- goal(E, X), remain_cust(E, Y, t), Y <= X. 

% -- assign edge to each segment
1 {as(D,(L,L'),0..C,t) : edge((L,L'))} 1 :- agent(D), capacity(D,C), at(D, L, t-1).

remain_cust(E,N-P,t)  :- remain_cust(E,N,t-1), P=#sum{W,D: as(D,E,W,t)}.

remain_loc(L,N-P,t)   :- remain_loc(L,N,t-1), P=#sum{W,L': as(_,(L, L'),W,t)}.

at(D, L', t) :- at(D, L, t-1), as(D,(L,L'),_,t).

% total delivered customers can not excess demand

:- (R+1) #sum{W,D:as(D,E,W,t)}, edge(E), remain_cust(E,R,t-1).

assigned(L, t) :- vertiport(L), at(_, L, t-1), as(_, (L, L'), W, t), W > 0.

:- vertiport(L), remain_loc(L, N, t-1), N > 0, not assigned(L, t).   

% stationed aircraft does not carry customers
:- as(_,(V,V),W,t), W!=0.

% Prioritize scheduling higher-revenue flight
#heuristic as(_,_,W,_). [W,true]
 
#program check(t).

:- query(t), goal(E, X), remain_cust(E, Y, t), Y > X.
 
% :- query(t), goal(E, _), not hold(E, t).

% #show seq/1. 
% #show as/4. 
% #show remain_cust/3. 
% #show remain_loc/3. 
% #show time/1. 
% #show goal/2.