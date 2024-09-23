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
max_weight(4).
weight(0).
weight(W+1) :- max_weight(MW), weight(W), W < MW.

%horizon in minutes
horizon(360).

% generate segment ground atoms unit.
#const max_segment=27.
maxspan(27).
seg(0).
seg(X+1) :- seg(X), X < max_segment.

% ----- GENERATE ROUTING

% assign arc to agent-segment
1 {as(D, (V, V'), X): edge(V, V')} 1 :- agent(D), seg(X), X < max_segment.
% constraint: segment X and segment X+1 of agent D must have assignments that are two consecutive arcs
:- as(D, (_, V'), X), as(D, (V'', _), X+1), V' != V''.
% assignment of segment 0 of agent D must have arc that its origin is initial location of agent D
:- init_loc(D, V), as(D, (V', _), 0 ), V != V'.
% assign request to each segment assignment. 
1 {as_rq(D, N, X): request(N, V, V', _, _); not_as_rq(D, X)} 1 :- as(D, (V, V'), X).
% constraint: only request that at segment time can be assigned to the segment
&diff {start_v(D, V, X) - 0} = T :- as_rq(D, N, X), request(N, V, V', _, T).
% no agent fly without passengers.
:- C = #count {D: as_rq(D, N, _)}, request(N, _, _, W, _), C*4 - W >= 4.


