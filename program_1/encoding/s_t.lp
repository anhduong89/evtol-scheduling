% ------ INSTANCES
% #include "instances/network_NY_0.lp".
% #include "instances/init.lp".
% #include "instances/rq.lp".
#const period=3.
#const max_seg=5.
max_seg(D,max_seg) :- agent(D).
seg(1..max_seg).
time(1..30).

% ------ WAIT TIME AND PICK UP POSSIBILITY
wait_time(E,T*5/10) :- flight_time(E,T).
rq_pkup_time(R,T) :- request(R,E,_,RT), time(T), wait_time(E,WT), T-WT <= RT*period <= T+WT.

% ------ GENERATE ROUTING

% -- assign edge to each segment
1 {as(D,E,X):edge(E), time(T)} 1 :- agent(D), X=start_seg..max_seg.

as(D,E,X,FT+CT+CTE) :- FT={FT':flight_time(E,FT'), as(D,E,X'), X'<=X}, CT={CT':charge_time(E,CT'), as(D,E,X'), X'<=X}, charge_time(E,CTE). 

% assign rq to the assigned flight
{as_rq(D,R):rq_pkup_time(R,T)} :- as(D,E,X).

% start vertiport as specicfied
:- init_loc(D,V), as(D,(V',_),_,1), V!=V'.

% only assign request once
:- request(R,_,_,_), as_rq(D,R), as_rq(D',R), D!=D'.

% consecutive trips have to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.

% consecutive trips have to have time connected
:- as(D,E,T,X), as(D,E',T',X+1), flight_time(E,FT), charge_time(E',CT), T+FT+CT<=T'.

%%% :~ as(D,E,W,_), as(_,E,W',_), W>0, W'>0, capacity(D,C), W<C.   [(D,E,W)@1]

% :~ as(D,E,W,X), as(_,E,W',X'), W>0, W'>0, X<X', capacity(D,C), W<C.   [1@1,D,E,W,X]

% :~ as(_,E,0,_), not edge_loop(E). [1@1]

% Prioritize scheduling higher-revenue flight


#show.
#show as/4.
#show as_rq/2.
#show rq_pkup_time/2.