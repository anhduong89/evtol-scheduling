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
horizon(71).

% generate segment ground atoms unit.

seg(0).
seg(X+1) :- seg(X), X < max_seg.

% ----- GENERATE ROUTING

% assign max segment to each 
% {max_seg(D, max_seg); max_seg(D, max_seg+1); max_seg(D, max_seg+2); max_seg(D, max_seg+3)} = 1 :- agent(D).

max_seg(D,max_seg):-agent(D).

%assign trip to segment: assigning trip (V, V1) if edge(V, V1) has passengers then it's the shortest trip in the reach of agent D at time step X

% 1 {as(D, (V, V1), X+1): distance((V, V1), S'), S' >= MS;as(D, (V, V), X+1)} 1:- seg(X+1), agent(D), X < M, max_seg(D, M), MS= #max {S:distance((V,V'),S), not no_cust(D,(V,V'),_), S>0}, as(D, (_, V),X).

1 {as(D, (V, V1), X): distance((V,V1),S'), S' >= MS;as(D,(V,V), X):edge(V,V)} 1:- seg(X), agent(D), X < M, max_seg(D, M), MS= #max {S:distance((V,V'),S), no_cust(D,(V,V'),Y), seg(Y), Y >= X,S>0}.

% assign trip for initial segment
1{as(D, (V, V'), 0): distance((V, V'), S'), S' >= Smin}1:- agent(D), init_loc(D, V) , Smin = #max {S: distance((V, V1), S), S > 0}.

:- init_loc(D,V), as(D,(V',_),0), V!=V'.

% consecutive trip need to be connected
:-as(D,(V,V1),X),as(D,(V2,V3),X+1),V1!=V2.

:-as(D,(V,V1),X),as(D,(V,V2),X),V1!=V2.


% 1 {as(D, (V, V1), X+1): distance((V, V1), S'), S' >= MS} 1:- seg(X+1), agent(D), X < M, max_seg(D, M), MS= #max {S:distance((V, V'), S), not no_cust(D,(V, V'),X), S > 0}, as(D, (_, V), X).


#heuristic as(D,(V,V),X) : not as_w(D,_,X,_). [1,sign]

#heuristic as_w(D,E,X,W) : as(D,E,X), not as_w(_,E,_,W), weight(W). [1, sign]

% at time step X of agent D, system choose whether has demand passengers or not.
%% can we use only one atom no_cust
% 0 {cust(D,(V,V'),X)} 1 :- agent(D), seg(X), edge(V,V').
% 0 {no_cust(D,(V,V'),X)} 1 :- agent(D), seg(X), edge(V,V').

% :- no_cust(D,(V,V'),X), cust(D,(V,V'),Y), Y>=X.

% :- not as_w(_,E,_,10), no_cust(_,E,_).
% term no_cust(E) if exist the 3rd flight on edge E




1 {no_cust(D,(V,V'),X):seg(X), as(D',(_,V),X)} 1:- as_w(D',(V, V'),Y,10), agent(D).

% &diff{arrival(D,V,Y) - no_cust(V,V')} > 0 :- no_cust(D,(V,V'),X), as(D,(_,V),Y), Y >= X.



#show no_cust/3.
#show cust/3.