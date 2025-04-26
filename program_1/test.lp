#include "instances/network_NY_0.lp".
#include "instances/init.lp".
total(T, E) :- edge(E), T = #sum{W,D,X : as(D,E,W,X)}. 
wasted(F) :- F = #count{(L,L'),D,X : as(D,(L,L'), 0, X), L!=L'}. 

flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.

total_time_need(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 

distance_time(X - Y) :- X = #max{T : flight_time_need(_, T)}, Y = #min{T : flight_time_need(_, T)}.

distance_charge(X - Y) :- X = #max{T : charge_time_need(_, T)}, Y = #min{T : charge_time_need(_, T)}.

#show total_time_need/2.