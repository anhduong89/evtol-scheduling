

edge(E) :- as(_,E,_,_). 
agent(D) :- as(D,_,_,_). 
step(D,S) :- as(D,_,_,S). 


total(T, E) :- edge(E), T = #sum{W,D,X : as(D,E,W,X)}.

% empty flights
wasted(F) :- F = #count{(L,L'),D,X : as(D,(L,L'), 0, X), L!=L'}.

flight_time_need(D, TD) :- agent(D), TD = #sum{T, X : flight_time(E, T), as(D, E, _, X)}.
charge_time_need(D, TD) :- agent(D), TD = #sum{T, X : charge_time((L,L1), T), as(D, (L,L1), _, X), as(D, (_,L), _, X-1)}.

total_time_need(D, FT + CT) :- agent(D), flight_time_need(D, FT), charge_time_need(D, CT). 

distance_time(X - Y) :- X = #max{T : flight_time_need(_, T)}, Y = #min{T : flight_time_need(_, T)}.

distance_charge(X - Y) :- X = #max{T : charge_time_need(_, T)}, Y = #min{T : charge_time_need(_, T)}.

solution_time(X) :- X = #sum{FT,D : flight_time_need(D, FT)}.

solution_charge(X) :- X = #sum{FT,D : charge_time_need(D, FT)}.


% min_time(D) :-  agent(D), total_time_need(D, X),  X = #min{T : total_time_need(_, T)}. 
% max_time(D) :-  agent(D), total_time_need(D, X),  X = #max{T : total_time_need(_, T)}.
agent_need_switch(D) :- total_time_need(D,T), T>horizon.
agent_to_switch(D) :- total_time_need(D,T), T< horizon.

number_of_agent_need_switch(C) :- C=#count{D0:agent_need_switch(D0)}.
% 1{max_time(D):max_time_set(D)}1.

% flight time and charging time until step S
flight_time_s(D, S, FT+CT) :- agent(D), step(D, S), 
                           FT = #sum{T, S': flight_time(E, T), as(D, E, _, S'), S' <= S},
                           CT = #sum{T, S': charge_time(E, T), as(D, E, _, S'), 1 < S' <= S}.

charging_time(CT) :- CT = #sum{T, E : charge_time(E, T)}.

shared(D, D', L',S, S', FT, FT', NFT-FT, NFT'-FT') :-
      agent_to_switch(D),
      agent_need_switch(D'),
      as(D, (_,L'), _, S),
      as(D', (_,L'), _, S'),
      flight_time_s(D, S, FT),
      total_time_need(D, NFT),
      flight_time_s(D', S', FT'),
      total_time_need(D', NFT').

mixed(D, D', L, S, S', FT1 + NFT2, FT2 + NFT1, X) :- shared(D, D', L, S, S', FT1, FT2 , NFT1,  NFT2), X = FT1 + NFT2 - FT2 - NFT1, X >= 0.

mixed(D, D', L, S, S', FT1 + NFT2, FT2 + NFT1, -X) :- shared(D, D', L, S, S', FT1, FT2 , NFT1,  NFT2), X = FT1 + NFT2 - FT2 - NFT1, X < 0.

% cut at segment that could provide better trajectory
% mixed_best_set(D, D', L, S, S', X, Y) :- mixed(D, D', L, S, S', X, Y, Z), Z = #min{Z' :  mixed(_, _, _, _, _, _, _, Z')}. 
mixed_best_set(D, D', L, S, S', X, Y) :- mixed(D, D', L, S, S', X, Y, Z), X<180, Y<180. 


C { mixed_best(D, D', L, S, S', X, Y) : mixed_best_set(D, D', L, S, S', X, Y) } C :- number_of_agent_need_switch(C).

:- mixed_best(D,D0,_,_,_,_,_), mixed_best(D,D1,_,_,_,_,_), D0!=D1.
:- mixed_best(D0,D,_,_,_,_,_), mixed_best(D1,D,_,_,_,_,_), D0!=D1.
:- mixed_best(D',D,L,_,_,_,_), mixed_best(D',D,L',_,_,_,_), L!=L'.
:- mixed_best(D',D,L,S,_,_,_), mixed_best(D',D,L,S',_,_,_), S!=S'.
:- mixed_best(D',D,L,S,S0,_,_), mixed_best(D',D,L,S,S1,_,_), S0!=S1.

no_swap(D) :- not mixed_best(D,_,_,_,_,_,_), agent_to_switch(D).
no_swap(D) :- not mixed_best(_,D,_,_,_,_,_), agent_need_switch(D).

nas(D,E,W,S) :- no_swap(D), as(D,E,W,S).

nas(D, E, W, S) :- as(D, E, W, S), mixed_best(D, _, _, S1, _, _, _), S <= S1.

nas(D, E, W, S - S2 + S1) :- as(D', E, W, S), mixed_best(D, D', _, S1, S2, _, _), S > S2.

nas(D, E, W, S) :- as(D, E, W, S), mixed_best(_, D, _, _, S2, _, _), S <= S2.

nas(D, E, W, S - S1 + S2) :- as(D', E, W, S), mixed_best(D', D, _, S1, S2, _, _), S > S1.




% #maximize { 1,D,D' : mixed_best(D,D',_,_,_,_,_) }.

#show.
% #show as(D,E,W,X): as(D,E,W,X), D=(31;16).
#show as(D,E,W,X): nas(D,E,W,X).
% #show nas/4.
% #show mixed(A,B,C,D,E,F,G,H) : mixed(A,B,C,D,E,F,G,H), A=31, B=16.
#show mixed_best(A,B,C,D,E,F,G) : mixed_best(A,B,C,D,E,F,G).
% #show mixed_best_set/7.
#show number_of_agent_need_switch/1.
% #show shared/9.
#show no_swap/1.