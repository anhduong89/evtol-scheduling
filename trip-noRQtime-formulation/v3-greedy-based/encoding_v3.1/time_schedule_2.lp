% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {0-start(D,(V,V'), 0)} <= -Ct :-    as(D, (V, V'), 0), charge_time((V, V'), Ct).

% start time at vertiport X+1 - start time vertiport X = flight time (X, X+1) + charge time at vertiport X+1 
&diff {start(D,E',X+1) - start(D,E,X)} >= T :-    flight_time(E,Ft), charge_time(E',Ct), T=Ft+Ct, 
                                                    as(D,E,X), as(D,E',X+1).

&diff {start(D,E,X) - no_cust(D,E1,X)} > 0 :-     as_w(D,E,X,_), no_cust(D,E1,X), 
                                                    flight_time(E,S), flight_time(E1,S1), S > S1.



% &diff {start(D, (V, V'), X) - start_v(D, V, X)} = 0 :- as(D, (V, V'), X).

% by segment X, arrival time at vertiport V' = start time at vertiport V + flight time
% &diff {arrival_v(D, V', X) - start_v(D, V, X)} = Ft :- distance((V, V'), S), Ft = (S*60)/speed, as(D, (V, V'), X).

%ready_v is the presumed start time of agent D at vertiport V at segment X: this compute by a summation of arrival_v(D, V, X-1) and a scalar value k.
% &diff {ready_v(D, V', X+1) - arrival_v(D, V', X)} >= 0 :- as(D, (V, V'), X).



% ------ DELIVERY SCHEDULE
% weight(0).
% weight(W+1) :- weight(W), W < capacity.
weight(4).
weight(8).
weight(10).

:- as_w(_,(V,V'),_,8), not as_w(_,(V,V'),_,4).

:- as_w(_,(V,V'),_,10), not as_w(_,(V,V'),_,8).

% as_w(D, (V, V), X, 0) :- as(D, (V, V), X).
0 {as_w(D,(V,V'),X,W): as(D,(V,V'),X),V!=V'} 1 :- weight(W), edge(V,V').

:- as_w(D,E,X,W), as_w(D,E,X,W'), W!=W'.

%
&diff {start(D',E,Y) - start(D,E,X)} > 0 :- as_w(D,E,X,W), as_w(D',E,Y,W'), W<W'.

%---- restrict the time of last trip not excess the horizon.
% &diff {start_v(D, V, X)-0} <= H-Ft :- as(D, (V, V'), X), seg(X), distance((V, V'), S), Ft = S/speed, horizon(H).

&diff {start(D,E,max_seg)-0} <= T :-  as(D,E,max_seg), 
                                        horizon(H), flight_time(E, Ft),
                                        T = H-Ft.

% &diff {arrival_v(D, V', M)-0} <= H :- as(D, (V, V'), M), max_seg(D, M), horizon(H).