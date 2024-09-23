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
#const total_timesteps=11.

distance_t((V, V'), ((S/(speed*30))+1)*2) :- distance((V, V'), S), V != V'.
distance_t((V, V), 1) :- edge(V, V).%horizon in minutes

horizon(360).
time(0..total_timesteps).
% generate segment ground atoms unit.
maxspan(10).

node_w(V, W, 0) :- W=#count{D: init_loc(D, V)}.
agent_pos(D, V, 0) :- init_loc(D, V).
max_weight(C*4) :- total_agent(C).  

weight(4).
weight(W+4) :- weight(W), max_weight(WM), W < WM.  

% init node weight
node_w(V, C*4, 0) :- C = #count{D: init_loc(D, V)}, vertiport(V).

% ----- GENERATE ROUTING



%if node V has available agent then we can assign agents to any arc (V, V') start from V
0 {edge_w((V, V'), W, T): weight(W), W <= WV} 1 :- edge(V, V'), time(T), vertiport(V), node_w(V, WV, T), WV > 0.

% total agents depart from V at time step T must not excess the available agents at (V, T).
:- node_w(V, WV, T), Out = #sum {WE, V': edge_w((V, V'), WE, T)}, Out != WV.

% available agents at vertiport V at time step T is the total of inbound agents at previous time step.
node_w(V, I, T) :- I = #sum{WI, VI: edge_w((VI, V), WI, T-S), distance_t((VI, V), S)}, vertiport(V), time(T), T > 0.


1 {agent_move(D, (V, V'), T): edge(V, V')} 1 :- agent_loc(D, V, T).

agent_loc(D, V', T+1) :- agent_move(D, (_, V'), T).

:- edge_w((V, V'), W, T), C = #count{D: agent_move(D, (V, V'), T)}, C!=W.