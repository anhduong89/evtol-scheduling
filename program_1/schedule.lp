% ------ INSTANCES
#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

% ------ GENERATE ROUTING
seg(0..max_seg).
max_seg(D,max_seg-1):-agent(D).

% -- assign edge to each segment
1 {as(D,E,0..C,X):edge(E)} 1 :- agent(D), capacity(D,C), X=0..(max_seg-1).

% total delivered customers can not excess demand
:- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

:- 1 #sum{W,D,X:as(D,E,W,X)}, edge(E), not request(_,E,_).

% stationed aircraft does not carry customers
:- as(_,(V,V),W,_), W!=0.

% start vertiport as specicfied
:- init_loc(D,V), as(D,(V',_),_,0), V!=V'.

% consecutive trips need to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.

% Prioritize scheduling higher-revenue flights earlier in the trajectory
#heuristic as(_,E,W,X):distance(E,S). [W*S*(max_seg-X),true]

%  ----------------------------
% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,E,0)} = 0 :- as(D,E,_,0).

% constraint the operation time of agent within horizon
&diff {start(D,E,M)} <= (horizon-Ft) :- as(D,E,_,M), max_seg(D,M), flight_time(E,Ft).

% arrival time specification
&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

% start time specification 
&diff {start(D,E',X+1) - start(D,E,X)} >= (FT+CT) :-  flight_time(E,FT), charge_time(E',CT), as(D,E,_,X), as(D,E',_,X+1).

%  ----------------------------
% ------ optimization: penalize empty flights.

% penalty(D, E, S, 1) :- as(D, E, 0, S).
% :~ penalty(D, _, S, P). [P@0, D, S]


%  ----------------------------

% display
#show as/4.