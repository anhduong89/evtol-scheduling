#include "instances/init.lp".

%  ----------------------------
{ch_time(D,0..60,X)} = 1:- as(D,E,_,X), distance(E,S), S>0.
ch_time(D,0,X) :- as(D,E,_,X), distance(E,0).



% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,1)} = 0 :- as(D,_,_,1).

&diff {b_before(D,1)} = B :- b_init(D,B).

&diff {b_after(D,X)-b_before(D,X+1)} >= FT :- charge_time(E,FT), as(D,E,_,X).

&diff {b_after(D,X)-b_before(D,X)} = CT :- ch_time(D,CT,X).

&diff {b_after(D,X)} <= EMAX :- emax(D,EMAX), agent(D), X=0..(max_seg+1).
&diff {0-b_after(D,X)} <= 0 :- agent(D), X=0..(max_seg+1).

&diff {0-b_before(D,X)} <= 0 :- agent(D), X=0..(max_seg+1).
&diff {b_before(D,X)} <= EMAX :- emax(D,EMAX), agent(D), X=0..(max_seg+1).


% &diff {arrival(D,M)} <= horizon :- as(D,_,_,M), max_seg(D,M).

&diff {arrival(D,X) - start(D,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

&diff {start(D,X+1) - arrival(D,X)} >= CT :- ch_time(D,CT,X+1), as(D,_,_,X), as(D,_,_,X+1).

#show ch_time/3.

