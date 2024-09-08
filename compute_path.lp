%9.1.2-planning
% gamma dollar per ton conversion for MER
% alpha dollar per mile to calculate fare, $/mile
% using Plana Aero RAM aircraft range 300miles; speed 185mph
% ---- CONSTANT
#const alpha=1. %  $/mile
#const dividealpha=2. % alpha = 0.5$/mile
#const gamma=185. % $/ton
#const speed=3. % 3mile per minute 185mph

#const dischg_rate=4. %kWh/mile
#const capacity = 4.
#const charging_rate = 250. %kW
#const divideCharging_rate = 4.
#const divideMER=100.

% how many minute aircraft need to be charged to fly 1 mile
% const minute_charged_per_mile = dischg_rate*60/250.
#const minute_charged_per_mile = 1.
% #const total_timesteps=11.

distance_t((V, V'), ((S/(speed*30))+1)*2) :- distance((V, V'), S), V != V'.
distance_t((V, V), 1) :- edge(V, V).%horizon in minutes

horizon(180).
time(0..total_steps).

% total number of agents
total_agent(C) :- C = #count{D: agent(D)}.

% 
max_weight(C*capacity) :- total_agent(C).
weight(capacity).
weight(W+4) :- weight(W), max_weight(WM), W < WM.

% init node weight
node_w(V, C*4, 0) :- C = #count{D: init_loc(D, V)}, vertiport(V).


% ----- GENERATE ROUTING
%if there are available agents in node V then we can assign them to any arc (V, V')
0 {edge_w((V, V'), W, T): weight(W), W <= WV} 1 :- edge(V, V'), time(T), vertiport(V), node_w(V, WV, T), WV > 0.

% number of agent that departs from V at time step T must not excess total available.
:- node_w(V, WV, T), Out = #sum {WE, V': edge_w((V, V'), WE, T)}, Out != WV.

% number of agents at V at step T is the sum of inbound agents to V from step T-S
node_w(V, I, T) :- I = #sum{WI, VI: edge_w((VI, V), WI, T-S), distance_t((VI, V), S)}, vertiport(V), time(T), T > 0.


