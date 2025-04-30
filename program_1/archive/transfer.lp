% #include "test_instance/raw_init.lp".
#include "instances/network_NY.lp".
% #include "test_instance/raw_remains_cust.lp".
% #include "schedule.lp".
#include "instances/init.lp".

% generate request and result:
vertiport(X):-edge((X, Y)).
vertiport(Y):-edge((X, Y)).

location(D, T, L):- dl(start(D,(L,_),_),T).
location(D, T, L):- dl(arrival(D,(_,L),_),T).
location(D, T, on_air):- dl(start(D,(_,_),S),T1), dl(arrival(D,(_,L),S),T2), T = (T1+1)..(T2-1).
location(D, T, L):- dl(arrival(D,(_,L),S),T1), dl(start(D,(L,_),S+1),T2), T = (T1)..(T2).

time(T) :- T = 0..180.

count(L, DC, T):- DC = #count {D:location(D, T, L)}, vertiport(L), time(T).

max_count(L, C):- C = #max{DC:count(L, DC, T)}, vertiport(L).


min_flight_time(L, FT) :- vertiport(L), FT = #min {FL:flight_time((L, Y), FL), L!=Y}.
drone(D):- dl(arrival(D, _, 10), _).
max_horizon(D, MH):- MH = #max {H:dl(arrival(D, (PL, L), 10), H)}, drone(D).
path(D, X, Y):-max_horizon(D, MH), dl(arrival(D, (X, Y), 10), MH).
result(D, L, MH):- max_horizon(D, MH), drone(D), path(D, PreL, L), charge_time((PreL, L), CT), min_flight_time(L, FT), MH <=(180-(CT+FT)).

% request(X, E, C):- remain_cust(E, C), C>0, X = 0.

agent(D):- result(D, _, _).
init_loc(D, L):- result(D, L, _).

capacity(D, 4):- agent(D).


% % ------ GENERATE ROUTING
% max_seg(D,max_seg):-agent(D).

% % -- assign edge to each segment
% 1 {as(D,E,0..C,X):edge(E)} 1 :- agent(D), capacity(D,C), X=min_seg..(max_seg).

% % total delivered customers can not excess demand
% :- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

% :- 1 #sum{W,D,X:as(D,E,W,X)}, edge(E), not request(_,E,_).

% % stationed aircraft does not carry customers
% :- as(_,(V,V),W,_), W!=0.

% % start vertiport as specicfied
% :- init_loc(D,V), as(D,(V',_),_,min_seg), V!=V'.

% % consecutive trips need to be connected
% :-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.




% #show result/3.
% #show request/3.
% #show capacity/2.
% #show agent/1.
% #show init_loc/2.
% #show as/4.

% % #show min_flight_time/2.
% % #show max_horizon/2.
% #show result/3.
% #show location/3.
% #show count/3.
#show max_count/2.