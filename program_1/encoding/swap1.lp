


edge(E) :- as(_,E,_,_). 
agent(D) :- as(D,_,_,_). 
step(D,S) :- as(D,_,_,S). 
% max_step_candidate(D,S) :- S = #max{S': step(D,S')}, agent(D).

% agent(0..33). 

total(T, E) :- edge(E), T = #sum{W,D,X : as(D,E,W,X)}.

% empty flights
wasted(F) :- F = #count{(L,L'),D,X : as(D,(L,L'), 0, X), L!=L'}.

flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.

operation_time(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 

distance_time(X - Y) :- X = #max{T : flight_time_need(_, T)}, Y = #min{T : flight_time_need(_, T)}.

distance_charge(X - Y) :- X = #max{T : charge_time_need(_, T)}, Y = #min{T : charge_time_need(_, T)}.

solution_time(X) :- X = #sum{FT,D : flight_time_need(D, FT)}.


min_time(D) :-  agent(D), operation_time(D, X),  X = #min{T : operation_time(_, T)}. 
sourcer(D) :-  agent(D), operation_time(D, X),  X = #max{T : operation_time(_, T)}. 


% flight time and charging time until step S
operation_time_s(D, S, FT+CT) :- agent(D), step(D, S), 
                           FT = #sum{T, S': flight_time(E, T), as(D, E, _, S'), S' <= S },
                           CT = #sum{T, S': charge_time(E, T), as(D, E, _, S'), 1 < S' <= S }.


cut_step(D,S) :- sourcer(D), operation_time_s(D,S,T), T>180.

min_cut_step_sourcer(D,S) :- S=#min{S',D:cut_step(D,S')}, sourcer(D).
max_step_candidate(D,S) :- S = #max{S',D: step(D,S')}, agent(D), sourcer(D'), D!=D'.
% max_step_sourcer(D,S) :- S=#max{S',D:step(D,S')}, sourcer(D).

swap_value(D,OT+(OT-OTS),OTS) :- min_cut_step_sourcer(D',S), operation_time(D',OT), operation_time_s(D',S-1,OTS), not connect_flights(D,_,_), max_step_candidate(D,S), operation_time(D,OT).

swap_value(D,OT1+(CT+FT)+(OT0-T0),T0) :- min_cut_step_sourcer(D0,S), operation_time(D0,OT0), operation_time_s(D0,S-1,T0), connect_flights(D,E,_), max_step_candidate(D,_), operation_time(D,OT1), charge_time(E,CT), flight_time(E,FT).


1 {best_swap_value(D): swap_value(D,T1,T2), T1 < 180, T2 < 180; no_swap} 1.

connect_flights(D,(V1,V),S+1) :- max_step_candidate(D,S), min_cut_step_sourcer(D',S'), sourcer(D'), as(D,(V0,V1),_,S), as(D',(V,V'),_,S'), V1!=V.

nas(D, E, W, X) :- as(D, E, W, X), sourcer(D1), D!= D1.

nas(D,E,W,X) :- as(D,E,W,X), sourcer(D), not cut_step(D,X).

nas(D,(V,V'),W',S1+S'-S2+1) :- sourcer(D'), cut_step(D',S'), as(D',(V,V'),W',S'), not connect_flights(D,_,_), max_step_candidate(D,S1), min_cut_step_sourcer(D',S2), best_swap_value(D).

nas(D,E,W,X) :- best_swap_value(D), as(D,E,W,X).

nas(D,(V,V'),W',S1+S'-S2+2) :- sourcer(D'), cut_step(D',S'), as(D',(V,V'),W',S'), max_step_candidate(D,S1), min_cut_step_sourcer(D',S2), connect_flights(D,_,_), best_swap_value(D).

nas(D,E,0,S) :- best_swap_value(D), connect_flights(E,S).



total_time_need_after_switch(D,FT+CT) :- agent(D),
                           FT = #sum{T, S': flight_time(E, T), nas(D, E, _, S')},
                           CT = #sum{T, S': charge_time(E, T), nas(D, E, _, S'), 1 < S'}.

% #show.
% % #show as(D,E,W,X): nas(D,E,W,X), min_time(D).
% % #show as(D,E,W,X): nas(D,E,W,X), sourcer(D).
% #show nas/4.
% #show min_time/1.
% #show sourcer/1.
% #show as(D,E,W,X) : as(D,E,W,X), min_time(D).
% #show as(D,E,W,X) : as(D,E,W,X), sourcer(D).
% % #show wasted/1.
% % #show shared/9.
% #show operation_time(D,T) : operation_time(D,T), min_time(D).
% #show operation_time(D,T) : operation_time(D,T), sourcer(D).
% #show solution_time/1.
% #show total_time_need_after_switch(D,T) : total_time_need_after_switch(D,T), min_time(D).
% #show total_time_need_after_switch(D,T) : total_time_need_after_switch(D,T), sourcer(D).
% #show cut_step/2.
% #show operation_time_s(D,S,T) : operation_time_s(D,S,T), min_time(D).
% #show operation_time_s(D,S,T) : operation_time_s(D,S,T), sourcer(D).
#show connect_flights(D,E,S) : connect_flights(D,E,S), D=32.
#show best_swap_value/1.
% #show swap_value/3.
#show min_cut_step_sourcer/2.
#show max_step_candidate(D,S) : max_step_candidate(D,S), D=1.
#show operation_time(D,T) : operation_time(D,T), D=(32;D'), sourcer(D').
#show operation_time_s(D,S-1,T) : operation_time_s(D,S-1,T), min_cut_step_sourcer(D,S).
#show no_swap/0.
#show swap_value(D,T1,T2): swap_value(D,T1,T2).
#show charge_time(E,T) : connect_flights(D,E,S), D=32, charge_time(E,T).
#show flight_time(E,T) : connect_flights(D,E,S), D=32, flight_time(E,T).