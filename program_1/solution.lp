% #show. 
% #show total/2. 
% #show wasted/1. 
% #show flight_time_need/2. 
% #show charge_time_need/2. 
% #show total_time_need/2. 
% #show distance_time/1. 
% #show distance_charge/1. 

edge(E) :- as(_,E,_,_). 

% agent(D) :- as(D,_,_,_). 

agent(0..33). 

total(T, E) :- edge(E), T = #sum{W,D,X : as(D,E,W,X)}. 
wasted(F) :- F = #count{(L,L'),D,X : as(D,(L,L'), 0, X), L!=L'}. 

flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.

total_time_need(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 

distance_time(X - Y) :- X = #max{T : flight_time_need(_, T)}, Y = #min{T : flight_time_need(_, T)}.

distance_charge(X - Y) :- X = #max{T : charge_time_need(_, T)}, Y = #min{T : charge_time_need(_, T)}.

solution_time(X) :- X = #sum{FT : flight_time_need(_, FT)}. 

solution_charge(X) :- X = #sum{FT : charge_time_need(_, FT)}. 

next(0, D) :- agent(D), flight_time_need(D, FT), FT = #min{T : flight_time_need(_, T)}.

next(X, D) :- next(X-1, D'), agent(D), flight_time_need(D', FT'), flight_time_need(D, FT), FT' < FT, #count{D1 : flight_time_need(D1, F1), FT' < F1, F1 < FT} = 0.

maxnext(X) :- X = #max{O : next(O, _)}.

1{pair(X, Y, D, D1) : next(Y, D1), X < Y, maxnext(Z), X + Y = Z} 1 :- next(X, D). 