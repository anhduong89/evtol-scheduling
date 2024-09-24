% ----- TIME SCHEDULE WITH DIFFERENT CONSTRAINT
% start time at initial vertiport = charge duration
&diff {start(D,E,0,W) - 0} >= 0 :-   as_w(D,E,0,W).

% start time at vertiport X+1 - start time vertiport X = flight time (X, X+1) + charge time at vertiport X+1 
&diff {start(D,E',X+1,W') - start(D,E,X,W)} >= T :-  flight_time(E,Ft), charge_time(E',Ct), T=Ft+Ct, 
                                                as_w(D,E,X,W), as_w(D,E',X+1,W').

% &diff {start(D,E,X) - no_cust(D,E1,X)} > 0 :-   as_w(D,E,X,_), no_cust(D,E1,X), 
%                                                 flight_time(E,S), flight_time(E1,S1), S > S1.

&diff {start(D,E,X,W) - start(D',E,Y,W')} > 0 :-  as_w(D,E,X,W), as_w(D',E,Y,W'), W>W'>0.

&diff {arrival(D,V',X) - start(D,(V,V'),X,W)} = Ft :- as_w(D,(V,V'),X,W), flight_time((V,V'),Ft).
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
weight(0).
:- as_w(_,(V,V'),_,8), not as_w(_,(V,V'),_,4).

:- as_w(_,(V,V'),_,10), not as_w(_,(V,V'),_,8).

% as_w(D, (V, V), X, 0) :- as(D, (V, V), X).
1 {as_w(D,E,X,W):weight(W)} 1 :- as(D,E,X).


% :- as_w(D,E,X,W), as_w(D,E,X,W'), W!=W'.
% :- Count = #count{E,W: as_w(_,E,_,W)}, weight(W), W!=0, Count > 1.

:- as_w(D,(V,V),X,W),W!=0.
:- as_w(D,E,X,W), as_w(D',E,Y,W), W!=0, D=D', X!=Y.
:- as_w(D,E,X,W), as_w(D',E,Y,W), W!=0, D!=D', X=Y.
:- as_w(D,E,X,W), as_w(D',E,Y,W), W!=0, D!=D', X!=Y.
%

%---- restrict the time of last trip not excess the horizon.
% &diff {start_v(D, V, X)-0} <= H-Ft :- as(D, (V, V'), X), seg(X), distance((V, V'), S), Ft = S/speed, horizon(H).

&diff {start(D,E,max_seg,W)-0} <= T :-  as_w(D,E,max_seg,W),
                                        horizon(H), flight_time(E, Ft),
                                        T = H-Ft.

% &diff {arrival_v(D, V', M)-0} <= H :- as(D, (V, V'), M), max_seg(D, M), horizon(H).