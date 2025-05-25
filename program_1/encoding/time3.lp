
gate(1..6,V) :- vertiport(V).
%  ----------------------------
{ch_time(D,(0..10)*2,X)} = 1:- as(D,E,_,X), distance(E,S), S>0.
ch_time(D,0,X) :- as(D,E,_,X), distance(E,0).

% -- assign a drone to a gate only
1 {as_gate((D,X),(G,V)): gate(G,V)} 1 :- as(D,(_,V),_,X).

% the order of gate's assignment of drone
1 {before(D,D',G); before(D',D,G)} 1 :- as_gate(D,G), as_gate(D',G), D != D'.

:- before(D,D',G), before(D',D'',G), before(D'',D,G).

% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,E,1)} = 0 :- as(D,E,_,1).

&diff {b_before(D,1)} = B :- b_init(D,B).

&diff {b_after(D,X)-b_before(D,X+1)} >= FT :- charge_time(E,FT), as(D,E,_,X).

&diff {b_after(D,X)-b_before(D,X)} = CT :- ch_time(D,CT,X).

&diff {b_after(D,X)} <= EMAX :- emax(D,EMAX), agent(D), X=0..(max_seg+1).
&diff {0-b_after(D,X)} <= 0 :- agent(D), X=0..(max_seg+1).

&diff {0-b_before(D,X)} <= 0 :- agent(D), X=0..(max_seg+1).
&diff {b_before(D,X)} <= EMAX :- emax(D,EMAX), agent(D), X=0..(max_seg+1).


&diff {arrival(D,E,M)} <= horizon :- as(D,E,_,M), max_seg(D,M).

&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

&diff {start(D,E',X+1) - arrival(D,E,X)} >= CT :- ch_time(D,CT,X+1), as(D,E,_,X), as(D,E',_,X+1).

% &diff {start(D,E,X+1) - arrival(D',E',Y)} <= 0 :- before((D,X),(D',Y),G), as(D,E,_,X+1), as(D',E',_,Y).

&diff {start(D,E,X+1) - arrival(D',E,Y)} <= 0 :- before((D,X),(D',Y),G), as(D,E,_,X+1), as(D',E',_,Y).

% &diff {start(D,X) - start(D,E,X)} = 0 :- as(D,E,_,X).

% &diff {arrival(D,X) - arrival(D,E,X)} = 0 :- as(D,E,_,X).