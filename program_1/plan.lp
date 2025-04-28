#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

max_seg(D,max_seg):-agent(D).
seg(1..max_seg).

% -- assign edge to each segment
1 {route(D,E,X):edge(E)} 1 :- agent(D), X=start_seg..max_seg.

% start vertiport as specicfied
:- init_loc(D,V), route(D,(V',_),1), V!=V'.

% consecutive trips need to be connected
:-route(D,(V,V1),X),route(D,(V2,V3),X+1),V1!=V2.

total_time(E, FT + CT) :- flight_time(E, FT), charge_time(E, CT). 

:~ agent(D), #sum{T, S : route(D, E, S), total_time(E, T)} > 180. [1@1, D] 

:~ agent(D), route(D, E, S), not edge_loop(E). [-1@2, D, E, S]
% #heuristic route(_,E,_):distance(E,S). [S,true]


% #show.
% #show route/4.
