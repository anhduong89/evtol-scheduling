% revenue for each agent D at route X
revenue(S*W*alpha/dividealpha, D, X) :- as_w(D, E, X, W), distance(E, S).

% % cost for each agent D at route X this was wrong
% emission_cost(S*minute_charged_per_mile*R, D, X) :- TS=#sum{S: as(D, E', X'), distance(E', S'), X' < X}, as(D, E, X), distance(E, S), mer(V, T, R), (TS*2/speed)-10 <= T <= (TS*2)/speed.

% an approximation; compute the lmp_mile atom for trip as(D, E, X) is simplified
% charging_cost((S*minute_charged_per_mile*LMPmile)/100, D, X) :- TS=#sum{S': as(D, E', X'), distance(E', S'), X' < X}, as(D, E, X), distance(E, S), lmp_mile(LMPmile, V, T), (TS*2)/speed-10 <= T <= (TS*2)/speed.



% find trajectory that that maximize profit
% #maximize{R@1, D, X:revenue(R, D, X)}.

% :- TR = #sum{R, D, X: revenue(R, D, X)}, TR > 2200. 

:~ revenue(R, D, X). [-R@1, D, X]
% % :~ charging_cost(C, D, X). [C@0, D, X]
#show.
#show as_w/4.
#show as/3.
#show passengers_served(Q, E) : Q = #sum{W,D,X:as_w(D, E, X, W)}, request(_, E, _).
% #show.
% #show violation(D1): as(D1, E', X'), as(D1, E1, X), as(D1, E2, X), E1 != E2.
% #show violation(D1): as(D1, E', X'), as(D1, (V, V1), X), as(D1, (V2, V3), X+1), V1 != V2.
% #show ch_time/3.


% #show total_em(EC) : EC = #sum{C, D, X: C=S*R, TS=#sum{S: as(D, E', X'), distance(E', S'), X' < X}, as(D, E, X), distance(E, S), mer(V, T, R), (TS*2/speed)-10 <= T <= (TS*2)/speed}.
% #show no_passengers/3.
% #show passengers/3.
% #show total_ch/1.
% #show total_em/1.
% #show passengers_served/2.