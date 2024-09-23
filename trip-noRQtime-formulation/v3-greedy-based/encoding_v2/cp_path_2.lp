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
horizon(180).

% generate segment ground atoms unit.

seg(0).
seg(X+1) :- seg(X), X < max_seg.

% ----- GENERATE ROUTING

% assign max segment to each 
% {max_seg(D, max_seg); max_seg(D, max_seg+1); max_seg(D, max_seg+2); max_seg(D, max_seg+3)} = 1 :- agent(D).

max_seg(D,max_seg):-agent(D).

%assign trip to segment: assigning trip (V, V1) if edge(V, V1) has passengers then it's the shortest trip in the reach of agent D at time step X

% 1 {as(D, (V, V1), X+1); as(D, (V, V), X+1)} 1:- seg(X+1), agent(D), max_number_of_seg(MX), X < MX, MS= #min {S:flight_time((V, V'), S), not no_passengers(D, (V, V'), X), S > 0}, as(D, (_, V), X), flight_time((V, V1), MS).
1{as(D,(V,V'),X+1):edge(V,V')}1:-seg(X+1),agent(D),X<M,max_seg(D,M).

% consecutive trip need to be connected
:-as(D,(V,V1),X),as(D,(V2,V3),X+1),V1!=V2.

:-as(D,(V,V1),X),as(D,(V,V2),X),V1!=V2.


% #heuristic 1 {as(D, (V, V1), X+1): flight_time((V, V1), S'), S' <= MS} 1 : seg(X+1), agent(D), X < M, max_seg(D, M), MS= #min {S:flight_time((V, V'), S), not no_passengers(D, (V, V'), X), S > 0}, as(D, (_, V), X).
#heuristic as(D,(V,V1),X+1) : flight_time((V,V1),MS), seg(X+1), agent(D),X<M, max_seg(D,M),MS=#max{S:flight_time((V,V'),S), not no_cust(D,(V,V'),X),S>0}, as(D,(_,V),X). [1,sign]

#heuristic as(D,(V,V1),X+1) : flight_time((V,V1),MS), seg(X+1), agent(D),X<M, max_seg(D,M),MS=#max{S:flight_time((V,V'),S), not no_cust(D,(V,V'),X),S>0}, as(D,(_,V),X). [1,sign]

% assign trip for initial segment
#heuristic as(D,(V,V'),0) : flight_time((V,V'),S'), S'<=Smin, agent(D), init_loc(D,V),Smin=#max{S:flight_time((V,V1),S), S>0}. [1,sign]

#heuristic as(D,(V,V),X) : not as_w(D,_,X,_). [1,sign]

#heuristic as_w(D,E,X,W) : as(D,E,X), not as_w(_,E,_,W), weight(W). [2, sign]

% at time step X of agent D, system choose whether has demand passengers or not.
1{cust(D,(V,V'),X);no_cust(D,(V,V'),X)}1 :- agent(D), seg(X), edge(V,V').

% 
&diff{no_cust(D,E,X) - start(D',E,Y)} = 1 :- as_w(D',E,Y,10), no_cust(D,E,X).

% &diff {arrival_v(D, V', X) - start_v(D, V, X)} = Ft :- flight_time((V, V'), S), Ft = (S*60)/speed, as(D, (V, V'), X).
% :- Wx = #sum{W, D: as_w(D, E, Y, W), Y < X}, as(D', E, X), request(_, E, Wr), Wx < Wr, no_passengers(D, E, X). 

