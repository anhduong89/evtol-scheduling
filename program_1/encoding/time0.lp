
#include "instances/network_NY_0.lp".
#const horizon=180.

max_seg(D, M) :- as(D, _, _, _), M = #max{X : as(D, _, _, X)}.
% % constraint
% % :- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

:- as(_,E,W,_), W > 0, edge_loop(E).

% % % start vertiport as specicfied
% % :- init_loc(D,V), as(D,(V',_),_,1), V!=V'.

% % consecutive trips need to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.

% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration

&diff {start(D,E,1)} = 0 :- as(D,E,_,1).

&diff {arrival(D,E,M)} <= horizon :- as(D,E,_,M), max_seg(D,M).

&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

&diff {start(D,E',X+1) - arrival(D,E,X)} >= CT :- charge_time(E',CT), as(D,E,_,X), as(D,E',_,X+1).


#show as/4.
#show dl/2.