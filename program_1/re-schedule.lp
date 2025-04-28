
nas(D, E, W, X) :- as(D, E, W, X), min_time(D0), max_time(D1), D != D0, D!= D1. 

nas(D, E, W, S) :- min_time(D), as(D, E, W, S), mixed_best(D, _, _, S1, _, _, _), S <= S1. 

nas(D, E, W, S - S2 + S1) :- min_time(D), as(D', E, W, S), mixed_best(D, D', _, S1, S2, _, _), S > S2. 

nas(D, E, W, S) :- max_time(D), as(D, E, W, S), mixed_best(_, D, _, _, S2, _, _), S <= S2. 

nas(D, E, W, S - S1 + S2) :- max_time(D), as(D', E, W, S), mixed_best(D', D, _, S1, S2, _, _), S > S1. 

#show. 
#show as(D, E, W, S) : nas(D, E, W, S). 

