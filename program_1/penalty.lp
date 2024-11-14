% ------ optimization: penalize empty flights.

penalty(D, E, S, 1) :- as(D, E, 0, S).
:~ penalty(D, _, S, P). [P@0, D, S]