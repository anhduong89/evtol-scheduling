% % revenue for each agent D at route X
% revenue(S*W*alpha/dividealpha, D, X) :- as_w(D, E, X, W), distance(E, S).
% revenue(W*S/2, N) :- C = #count {D: as_rq(D, N, _)}, request(N, V, V', W, _), C*4 - W > 0, distance((V, V'), S).
% revenue(C*4*S/2, N) :- C = #count {D: as_rq(D, N, _)}, request(N, V, V', W, _), C*4 - W <= 0, C != 0, distance((V, V'), S).
revenue(W, N) :- C = #count {D: as_rq(D, N, _)}, request(N, V, V', W, _), C*4 - W > 0.
revenue(C*4, N) :- C = #count {D: as_rq(D, N, _)}, request(N, V, V', W, _), C*4 - W <= 0, C != 0.
% % % cost for each agent D at route X
% emission_cost(S*minute_charged_per_mile*R, D, X) :- TS=#sum{S: as(D, E', X'), distance(E', S'), X' < X}, as(D, E, X), distance(E, S), mer(V, T, R), (TS*2/speed)-10 <= T <= (TS*2)/speed.

% charging_cost(S*minute_charged_per_mile*R, D, X) :- TS=#sum{S': as(D, E', X'), distance(E', S'), X' < X}, as(D, E, X), distance(E, S), lmp(V, T, R), (TS*2)/speed-10 <= T <= (TS*2)/speed.

% find trajectory that that maximize profit
#maximize{R:revenue(R, N)}.
% #minimize{C@0, D, X:emission_cost(C, D, X)}.
% #minimize{C@0, D, X:charging_cost(C, D, X)}.



% passengers_served(Q, E) :- Q = #sum{W,D,X:as_w(D, E, X, W)}, request(_, E, _).

% #show dl/2.
#show as/3.
% #show ch_time/3.
#show as_rq/3.
#show revenue/2.
% #show not_as_rq/2.
% #show no_passengers/3.
% #show passengers/3.

% #show passengers_served/2.