%9.1.2-planning
%Nathan formulation
% gamma dollar per ton conversion for MER

% alpha dollar per mile to calculate fare, $/mile
% using Plana Aero RAM aircraft range 300miles; speed 185mph 
% discharge rate 4kWh/mile
#const alpha=1. % round up from 0.5 $/mile
#const dividealpha=2. % any multiplication with alpha must divide to dividealpha
#const gamma=185. % $/ton
#const speed=100. % 3mile per minute 185mph

#const dischg_rate=4.
#const capacity = 4.
#const charging_rate = 250. %kW (per hour)
#const divideCharging_rate = 4.
#const divideMER=100.

% 1 min charge 250/60 kWh
% 1 min charged can flight 250/(60*4)
% #const minute_charged_per_mile = dischgRate [kWh/mile] * (60m/ charging_rate)= dischg_rate*60/250.
#const minute_charged_per_mile = 1. 

% 
% fly 1 mile <=> E_discharge kWh 
% lmp: $/kWh -> $/mile => lmp($/mile) = lmp/M

% weight atom, need to change if capacity is not homogeneous

% weight(1).
% weight(3).
%horizon in minutes
% horizon(180).

% revenue per mile
% generate segment ground atoms unit.

% seg(0).
% seg(X+1) :- seg(X), X < max_seg.
weight(4).
weight(3).
weight(2).
weight(1).
weight(0).

% ------ GENERATE ROUTING
% -- assign max segment to each 
% {max_seg(D, max_seg); max_seg(D, max_seg+1); max_seg(D, max_seg+2); max_seg(D, max_seg+3)} = 1 :- agent(D).

max_seg(D,max_seg):-agent(D).

% -- assign edge to each segment
% 1 {as(D,E,W,0):edge(E),weight(W)} 1 :- agent(D).
1 {as(D,E,W,X):edge(E),weight(W),as(D,_,_,X-1),X>=1;as(D,E',W',X):weight(W'),edge(E'),X=0} 1 :- agent(D), X=0..max_seg.


#heuristic as(_,E,4,_):distance(E,S). [4*S,true]
#heuristic as(_,E,3,_):distance(E,S). [3*S,true]
#heuristic as(_,E,2,_):distance(E,S). [2*S,true]
#heuristic as(_,E,1,_):distance(E,S). [S,true]
% #heuristic as(_,E,0,_),distance(E,S). [1,true]

% total delivered customers < 11 each edge not stationary
:- 11 #sum{W,D,X:as(D,E,W,X)}, edge(E).
:- as(_,(V,V),W,_), W!=0.
% start vertiport
:- init_loc(D,V), as(D,(V',_),_,0), V!=V'.
% consecutive trip need to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.

% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,E,0)} = 0 :- as(D,E,_,0).

&diff {start(D,E,M)} <= T :- as(D,E,_,M), max_seg(D,M), flight_time(E,Ft), T=horizon-Ft.

&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

% #maximize {W,D,E,X:as(D,E,W,X)}.
% revenue(TR) :- TR = #sum{R,D,X:R=W*S,as(D,E,W,X), distance(E,S)}.
#show as/4.
% #show revenue/1.

