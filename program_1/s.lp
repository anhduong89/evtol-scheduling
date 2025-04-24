% ------ INSTANCES
#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

max_seg(D,max_seg):-agent(D).
seg(1..max_seg).

% ------ GENERATE ROUTING

% -- assign edge to each segment
1 {as(D,E,0..C,X):edge(E)} 1 :- agent(D), capacity(D,C), X=start_seg..max_seg.

% total delivered customers can not excess demand
:- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

:- as(_,E,W,_), W > 0, edge_loop(E).

% start vertiport as specicfied
:- init_loc(D,V), as(D,(V',_),_,1), V!=V'.

% consecutive trips need to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.

%%% :~ as(D,E,W,_), as(_,E,W',_), W>0, W'>0, capacity(D,C), W<C.   [(D,E,W)@1]

:~ as(D,E,W,X), as(_,E,W',X'), W>0, W'>0, X < X', capacity(D,C), W<C.   [1@1,D,E,W,X]

:~ as(_,E,0,_), not edge_loop(E). [1@1]

% Prioritize scheduling higher-revenue flight
% #heuristic as(_,E,W,_):distance(E,S). [W*S,true]

% #heuristic as(_,_,W,_). [W,true]

#heuristic as(_,E,W,_):distance(E,S). [W*S,false]


#show.
#show as/4.
