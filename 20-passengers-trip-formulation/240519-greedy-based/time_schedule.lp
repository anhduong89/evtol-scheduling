% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {0-start_v(D, V, 0)} = -Ct :- as(D, (V, V'), 0), distance((V, V'), S), Ct = S*minute_charged_per_mile.


% start time at vertiport X+1 - start time vertiport X = flight time (X, X+1) + charge time at vertiport X+1 
&diff {start_v(D, V', X+1)-arrival_v(D, V', X)} = Ct :- distance((V', V''), S), Ft = (S/speed), as(D, (_, V'), X), as(D, (V', V''), X+1), Ct = S*minute_charged_per_mile.

% &diff {start(D, (V, V'), X) - start_v(D, V, X)} = 0 :- as(D, (V, V'), X).

% by segment X, arrival time at vertiport V' = start time at vertiport V + flight time
&diff {arrival_v(D, V', X) - start_v(D, V, X)} = Ft :- distance((V, V'), S), Ft = (S/speed), as(D, (V, V'), X).

%ready_v is the presumed start time of agent D at vertiport V at segment X: this compute by a summation of arrival_v(D, V, X-1) and a scalar value k.
&diff {ready_v(D, V', X+1) - arrival_v(D, V', X)} = 15 :- as(D, (V, V'), X).



% ------ DELIVERY SCHEDULE
% assign number of passengers
1 {as_w(D, (V, V'), X, W): weight(W)} 1 :- as(D, (V, V'), X), V!= V'.

as_w(D, (V, V), X, 0) :- as(D, (V, V), X).

% if agent D' pick up passenger after agent D and agent D carries less than its capacity then agent D will carry 0 passenger
&diff {start_v(D, V, X) - start_v(D', V, Y) } <= 0:- as_w(D, (V, V'), X, W), as_w(D', (V, V'), Y, W'), D!=D', capacity(D, C), capacity(D', C'), W < C, W' = 0.

% % if agent D' pick up passenger after agent D and agent D' carries less than its capacity then agent D will carry full capacity
% &diff {start(D, E, X) - start(D', E, Y) } <= 0:- as_w(D, E, X, W), as_w(D', E, Y, W'), D!=D', capacity(D, C), capacity(D', C'), W = C, W' <= C'.

% if agent D' pick up passenger after agent D and agent D carries less than its capacity then agent D will carry 0 passenger
&diff {start_v(D, V, X) - start_v(D', V, Y) } <= 0:- as_w(D, (V, V'), X, W), as_w(D', (V, V'), Y, W'), D!=D', capacity(D, C), capacity(D', C'), W = 0, W' = 0.

% if total passengers less than demand, then there are no agents carry less than their capacity. This help speed up the solving but limit on number of agent. Don't know y yet.
% :- as_w(D, E, M, W), capacity(D, C), W < C, TW = #sum{W1: as_w(_, E, _, W1)}, request(_, E, RW), TW < RW.

% total passengers pick up at route E not excess demand.
:- TW = #sum{W, D, X: as_w(D, (V, V'), X, W)}, request(_, (V, V'), RW), TW > RW.

% only one carry diff 0 and less than capacity
% :- A = #count{D, E, X: as_w(D, E, X, W), 0 < W < C, capacity(D, C)}, request(_, E, _), A > 1.


% restrict the time of last trip not excess the horizon.
&diff {start_v(D, V, X)-0} <= H-Ft :- as(D, (V, V'), X), seg(X), distance((V, V'), S), Ft = S/speed, horizon(H).
