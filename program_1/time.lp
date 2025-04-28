
max_seg(D, M) :- as(D, _, _, _), M = #max{X : as(D, _, _, X)}. 

% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration

&diff {start(D,E,1)} = 0 :- as(D,E,_,1).

&diff {arrival(D,E,M)} <= horizon :- as(D,E,_,M), max_seg(D,M).

&diff {arrival(D,E,X) - start(D,E,X)} = FT :- flight_time(E,FT), as(D,E,_,X).

&diff {start(D,E',X+1) - arrival(D,E,X)} >= CT :- charge_time(E',CT), as(D,E,_,X), as(D,E',_,X+1).

#show dl/2. 