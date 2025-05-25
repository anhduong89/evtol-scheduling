
agent(D) :- as(D,_,_,_).
gate(1..34,V) :- vertiport(V).
max_seg(D,M) :- M = #max{X:as(D,_,_,X)}, agent(D).

% -- assign a drone to a gate only
1 {as_gate((D,X),(G,V)): gate(G,V)} 1 :- as(D,(_,V),_,X).

% the order of gate's assignment of drone
1 {before(D,D',G); before(D',D,G)} 1 :- as_gate(D,G), as_gate(D',G), D != D'.

:- before(D,D',G), before(D',D'',G), before(D'',D,G).

% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,E,1)} >= 0 :- as(D,E,_,1).

&diff {arrival(D,E,M)} <= horizon :- as(D,E,_,M), max_seg(D,M).

&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

&diff {start(D,E',X+1) - arrival(D,E,X)} >= CT :- charge_time(E',CT), as(D,E,_,X), as(D,E',_,X+1).

&diff {start(D,E,X+1) - arrival(D',E',Y)} <= 0 :- before((D,X),(D',Y),G), as(D,E,_,X+1), as(D',E',_,Y).

% &diff {start(D,X+1) - arrival(D',Y)} <= 0 :- before((D,X),(D',Y),G).

% &diff {start(D,X) - start(D,E,X)} = 0 :- as(D,E,_,X).

% &diff {arrival(D,X) - arrival(D,E,X)} = 0 :- as(D,E,_,X).