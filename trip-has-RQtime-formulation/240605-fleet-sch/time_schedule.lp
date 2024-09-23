% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% ----- time trajectory:(A, B, 0) (B, C, 1) arrival(D, A, 0) -> start(D, A, 0) -> arrival(D, B 1) -> start(D, B, 1)
%-- start(D, 0) = charge_time for upcoming flight.
% if agent flight
&diff {start_v(D, V, 0)-0} = T :- as(D, (V, V'), 0), distance((V, V'), S), Ct = S*minute_charged_per_mile, T = (Ct/30)+1, S > 0.
% if agent stationed
&diff {start_v(D, V, 0)-0} = 0 :- as(D, (V, V), 0).

&diff {arrival_v(D, V, 0)-0} = 0 :- init_loc(D, V).

%-- as(D, V, V', X): start(D, V, X) -> arrival(D, V', X+1)
% start time at vertiport X+1 - start time vertiport X = flight time (X, X+1) + charge time at vertiport X+1 
&diff {arrival_v(D, V', X+1)-start_v(D, V, X)} = T :- distance((V, V'), S), Ft = (S/speed), as(D, (V, V'), X), Ct = S*minute_charged_per_mile, T = (Ct/30)+1, S>0.


% as(D, V, V', X): arrival(D, V, X) -> start(D, V, X) 
&diff {start_v(D, V, X)-arrival_v(D, V, X)} = T :- distance((V, V'), S), Ct = (S/speed), as(D, (V, V'), X), T = (Ct/30)+1, S > 0.

% if agent station at time step X  as(D, V, V, X): arrival_v(D, V, X) start(D, V, X)
&diff {arrival_v(D, V, X)-start_v(D, V, X)} = 0 :- as(D, (V, V), X).
&diff {start_v(D, V, X)-arrival_v(D, V, X+1)} = 0 :- as(D, (V, V), X).
% &diff {arrival_v(D, V', X) - start_v(D, V, X)} = 0 :- distance((V, V'), S), as(D, (V, V'), X), S = 0.

%ready_v is the presumed start time of agent D at vertiport V at segment X: this compute by a summation of arrival_v(D, V, X-1) and a scalar value k.
% &diff {ready_v(D, V', X+1) - arrival_v(D, V', X)} = 1 :- as(D, (V, V'), X).

% restrict the time of last trip not excess the horizon.
&diff {start_v(D, V, X)-0} <= T :- as(D, (V, V'), X), seg(X), distance((V, V'), S), Ft = S/speed, horizon(H), T = ((H-Ft)/30)+1.


% if agent is stationed at time X for ex: as(D, (V, V), X)

% &diff {arrival_v(D, V, X)-start_v(D, V, X)} = 0 :- as(D, (V, V), X).