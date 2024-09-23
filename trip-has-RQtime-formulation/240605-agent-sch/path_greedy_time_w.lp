%9.1.2-planning
%Nathan formulation
% gamma dollar per ton conversion for MER
% should look for two step forward but doing optimize single step.
% alpha dollar per mile to calculate fare, $/mile
% using Plana Aero RAM aircraft range 300miles; speed 185mph 
#const alpha=1. % round up from 0.5 $/mile
#const dividealpha=2. % any multiplication with alpha must divide to dividealpha
#const gamma=185. % $/ton
#const speed=3. % 3mile per minute 185mph

#const dischg_rate=4.
#const capacity = 4.
#const charging_rate = 250. %kW
#const divideCharging_rate = 4.
#const divideMER=100.

% #const minute_charged_per_mile = dischg_rate*60/250.
#const minute_charged_per_mile = 1.
% weight atom, need to change if capacity is not homogeneous
max_weight(10).
weight(0).
weight(W+1) :- max_weight(MW), weight(W), W < MW.

%horizon in minutes
horizon(360).

% generate segment ground atoms unit.
max_number_of_seg(10).
seg(0).
seg(X+1) :- seg(X), max_number_of_seg(M), X < M.

% ----- GENERATE ROUTING

%assign trip to segment: assigning trip (V, V1) if edge(V, V1) has passengers then it's the shortest trip in the reach of agent D at time step X

1 {as(D, (V, V1), X+1); as(D, (V, V), X+1)} 1 :- seg(X+1), agent(D), max_number_of_seg(MX), X < MX, MS= #min {S:distance((V, V'), S), not no_passengers(D, (V, V'), X+1), S > 0}, as(D, (_, V), X), distance((V, V1), MS).

% assign trip for initial segment
as(D, (V, V'), 0) :- agent(D), init_loc(D, V), distance((V, V'), Smin), Smin = #min {S: distance((V, V1), S), S > 0}.

% need to check as we already not allow this.
:- as(D, (V, V1), X), as(D, (V, V2), X), V1 != V2.

% at time step X of agent D, system choose whether has demand passengers or not.
1 {passengers(D, (V, V'), X);no_passengers(D, (V, V'), X)} 1:- agent(D), seg(X), edge(V, V').

% for agent D at beginning of segment X, we check all passenger allocations at segment Y that Y < X: if there is an assignment less than capacity, then we consider it as no_passengers.
&diff {ready_v(D, V, X) - start_v(D', V, Y)} > 0 :- as_w(D', (V, V'), Y, W'), capacity(D', C'), W' < C', no_passengers(D, (V, V'), X).

% :- Wx = #sum{W, D: as_w(D, E, Y, W), Y < X}, as(D', E, X), request(_, E, Wr), Wx < Wr, no_passengers(D, E, X). 