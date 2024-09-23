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
weight(0).
% weight(W+1) :- weight(W), W < capacity.
weight(4).
weight(2).
%horizon in minutes
horizon(185).

% generate segment ground atoms unit.

seg(0).
seg(X+1) :- seg(X), X < max_seg.

% ----- GENERATE ROUTING

% assign max segment to each 
% {max_seg(D, max_seg); max_seg(D, max_seg+1); max_seg(D, max_seg+2); max_seg(D, max_seg+3)} = 1 :- agent(D).

max_seg(D, max_seg) :- agent(D).

%assign trip to segment: assigning trip (V, V1) if edge(V, V1) has passengers then it's the shortest trip in the reach of agent D at time step X

% 1 {as(D, (V, V1), X+1); as(D, (V, V), X+1)} 1:- seg(X+1), agent(D), max_number_of_seg(MX), X < MX, MS= #min {S:distance((V, V'), S), not no_passengers(D, (V, V'), X), S > 0}, as(D, (_, V), X), distance((V, V1), MS).
1 {as(D, (V, V1), X+1): distance((V, V1), S'), S' <= MS} 1:- seg(X+1), agent(D), X < M, max_seg(D, M), MS= #min {S:distance((V, V'), S), not no_passengers(D, (V, V'), X), S > 0}, as(D, (_, V), X).

% assign trip for initial segment
1{as(D, (V, V'), 0): distance((V, V'), S'), S' <= Smin}1:- agent(D), init_loc(D, V) , Smin = #min {S: distance((V, V1), S), S > 0}.


% need to check as we already not allow this.
% :- as(D, (V, V1), X), as(D, (V, V2), X), V1 != V2.


% at time step X of agent D, system choose whether has demand passengers or not.
1 {passengers(D, (V, V'), X);no_passengers(D, (V, V'), X)} 1:- agent(D), seg(X), edge(V, V').

% for agent D at beginning of segment X, we check all passenger allocations at segment Y that Y < X: if there is an assignment less than capacity, then we consider it as no_passengers.
&diff {ready_v(D, V, X) - start_v(D', V, Y)} > 0 :- as_w(D', (V, V'), Y, W'), W' < capacity, no_passengers(D, (V, V'), X).

% :- Wx = #sum{W, D: as_w(D, E, Y, W), Y < X}, as(D', E, X), request(_, E, Wr), Wx < Wr, no_passengers(D, E, X). 