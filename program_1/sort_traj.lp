#include "instances/network_NY_0.lp".
#include "test_0.lp".
% edge(E) :- as(_,E,_,_). 
% agent(D) :- as(D,_,_,_). 
% step(D, S) :- as(D,_,_,S). 
% flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
% charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.
% total_time_need(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 
solution_time(X) :- X = #sum{FT,D : flight_time_need(D, FT)}.
% #show total_time_need/2.
% #show as/4.
#show solution_time/1.
% #show flight_time_need/2.

