% #show. 
% #show total/2. 
% #show wasted/1. 
% #show flight_time_need/2. 
% #show charge_time_need/2. 
% #show total_time_need/2. 
% #show distance_time/1. 
% #show distance_charge/1. 

edge(E) :- as(_,E,_,_). 

agent(D) :- as(D,_,_,_). 

step(S) :- as(_,_,_,S). 


% agent(0..33). 


total(T, E) :- edge(E), T = #sum{W,D,X : as(D,E,W,X)}. 
wasted(F) :- F = #count{(L,L'),D,X : as(D,(L,L'), 0, X), L!=L'}. 

flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.

total_time_need(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 

distance_time(X - Y) :- X = #max{T : flight_time_need(_, T)}, Y = #min{T : flight_time_need(_, T)}.

distance_charge(X - Y) :- X = #max{T : charge_time_need(_, T)}, Y = #min{T : charge_time_need(_, T)}.

solution_time(X) :- X = #sum{FT : flight_time_need(_, FT)}. 

solution_charge(X) :- X = #sum{FT : charge_time_need(_, FT)}. 


min_time(D) :-  agent(D), total_time_need(D, X),  X = #min{T : total_time_need(_, T)}. 
max_time(D) :-  agent(D), total_time_need(D, X),  X = #max{T : total_time_need(_, T)}. 

flight_time_s(D, S, FT+CT) :- agent(D), step(S), 
                           FT = #sum{T, S': flight_time(E, T), as(D, E, _, S'), S' < S }, 
                           CT = #sum{T, S': charge_time(E, T), as(D, E, _, S'), S' < S }. 
                            
charging_time(CT) :- CT = #sum{T, E : charge_time(E, T)}.

shared(D, D', L',S, S', FT, FT', NFT-FT, NFT'-FT') :-
      min_time(D), 
      max_time(D'), 
      as(D, (_,L'), _, S), 
      as(D', (_,L'), _, S'),
      flight_time_s(D, S, FT), 
      total_time_need(D, NFT), 
      flight_time_s(D', S', FT'),
      total_time_need(D', NFT').

mixed(D, D', L, S, S', FT1 + NFT2, FT2 + NFT1, X) :- shared(D, D', L, S, S', FT1, FT2 , NFT1,  NFT2), X = FT1 + NFT2 - FT2 - NFT1, X >= 0. 

mixed(D, D', L, S, S', FT1 + NFT2, FT2 + NFT1, -X) :- shared(D, D', L, S, S', FT1, FT2 , NFT1,  NFT2), X = FT1 + NFT2 - FT2 - NFT1, X < 0. 


mixed_best(D, D', L, S, S', X, Y) :- mixed(D, D', L, S, S', X, Y, Z), Z = #min{Z' :  mixed(_, _, _, _, _, _, _, Z') }. 

nas(D, E, W, X) :- as(D, E, W, X), min_time(D0), max_time(D1), D != D0, D!= D1. 

nas(D, E, W, S) :- min_time(D), as(D, E, W, S), mixed_best(D, _, _, S1, _, _, _), S <= S1. 

nas(D, E, W, S - S2 + S1) :- min_time(D), as(D', E, W, S), mixed_best(D, D', _, S1, S2, _, _), S > S2. 

nas(D, E, W, S) :- max_time(D), as(D, E, W, S), mixed_best(_, D, _, _, S2, _, _), S <= S2. 

nas(D, E, W, S - S1 + S2) :- max_time(D), as(D', E, W, S), mixed_best(D', D, _, S1, S2, _, _), S > S1. 

#show. 
#show as(D, E, W, S) : nas(D, E, W, S). 

