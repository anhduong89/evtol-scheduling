
% ----- ASSIGN CHARGE DURATION
% energy level converted to energy-flight time( minutes that agents are able to be on the sky)
time(0..12).
#const time_ratio=5.
% % % assign charging time for each trip assigmment
1 {ch_time(D, X, Ct) : time(Ct)} 1 :- as(D, (V, V'), X), V!= V'.

% if assign a null edge then agent will not charge.
ch_time(D, X, 0) :- as(D, (V, V), X).

%
ch_time(D, M, 0) :- max_number_of_seg(M), agent(D).
% total charge time should not excess the demanding flight energy
% :- TS = #sum{S: as(D, E, X), distance(E, S)}, TCt = #sum {Ct, X: ch_time(D, X, Ct)}, agent(D), TCt != ((TS-T0)/speed)*6, init_energy(D, T0, 0).

% convert battery to energy_time with time 
init_energy(D, B/(DCR*speed)) :- init_battery(D, B), dischg_rate(D, DCR).

% energy before charge at initial vertiport
&diff {beforeCharge(D, 0)-0} = T :- init_energy(D, T).

% energy after charge is equal to sum of energy before charge & energy gain from charge
&diff {afterCharge(D,X)-beforeCharge(D, X)} = Ct*time_ratio :- ch_time(D, X, Ct).

% energy after a flight (before charge at segment X+1) is equal to the deduction of flight discharge from current energy(which is after charge at segment X)
&diff { afterCharge(D, X)-beforeCharge(D, X+1)} = Ft :- as(D, E, X), distance(E, S), Ft = (S/speed).

% energy before charge and after charge must not excess roof and floor energy
&diff {afterCharge(D, X)-0} >= 0 :- seg(X), agent(D).
&diff {beforeCharge(D, X)-0} >= 0 :- seg(X), agent(D).
&diff {afterCharge(D, X)-0} <= 60 :- seg(X), agent(D).
&diff {beforeCharge(D, X)-0} <= 60 :- seg(X), agent(D).



% compute the cost regard the trip assignment and charging time, lmp_mer will be average value of give segment.
% cost(D, C) :- TCt = #sum{Ct*RATE, X:as(D, (V, _), X), ch_time(D, X, Ct), lmp_mer(V, X, RATE)}, agent(D), C = TCt/6.
cost(Ct*time_ratio*R, D, X) :- as(D, _, X), ch_time(D, X, Ct), lmp_mer(V, X, R).

% #minimize{C,D, X: cost(D, C, X)}.

