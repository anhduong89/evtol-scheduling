#const max_seg=3.  

#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

at(D, L, 0)      :- init_loc(D, L). 

remain_cust(E, R, 0) :- request(_,E, R). 

remain_loc(L, R, 0)  :- vertiport(L), R = #sum{P, L' : request(_, (L, L'), P)}. 

goal(E, 27) :- edge(E). 

step(1..max_seg).

% #program step(t).

% step(S). 

% -- assign edge to each segment
1 {as(D,(L,L'),0..C,S):edge((L,L'))} 1 :- agent(D), capacity(D,C), step(S), at(D, L, S-1).

% :- step(S), remain_cust(E, R, S-1), R > 0, as(D,E,0,S).

remain_cust(E,N-P,S) :- remain_cust(E,N,S-1), P=#sum{W,D: as(D,E,W,S)}, step(S).

remain_loc(L, N-P,S) :- vertiport(L), remain_loc(L, N,S-1), P=#sum{W,L': as(_,(L, L'),W,S)}, step(S).

at(D, L', S) :- step(S), at(D, L, S-1), as(D,(L,L'),_,S).

% total delivered customers can not excess demand
:- (R+1) #sum{W,D:as(D,E,W,S)}, edge(E), remain_cust(E,R,S-1), step(S).

assigned(L, S) :- step(S), vertiport(L), at(_, L, S-1), as(_, (L, L'), W, S), W > 0.

:- step(S), vertiport(L), remain_loc(L, N, S-1), N > 0, not assigned(L, S).   

% :- step(S), vertiport(L), at(D, L, S-1), P=#sum{W,L': as(_,(L, L'),W,S)} 0. 


% stationed aircraft does not carry customers
:- as(_,(V,V),W,t), W!=0.


% Prioritize scheduling higher-revenue flight
% #heuristic as(_,E,W,S) : remain_cust(E,R,S-1). [W*R,true]


% #heuristic as(_,_,W,_). [W,true]


% #program check(S).

% false(E, X, Y):- edge(E), goal(E, X), remain_cust(E, Y, max_seg), Y  > X.

:- edge(E), goal(E, X), remain_cust(E, Y, max_seg), Y  > X.

% #show step/1. 
% #show as/4. 
% #show remain_cust/3. 
% #show at/3. 
% #show false/3. 