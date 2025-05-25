% ------ optimization_2.1: penalize empty flights and not stationed. (could has better solution)

penalty(D, E, S, 1) :- as(D, E, 0, S), not edge_loop(E).
:~ penalty(D, _, S, P). [P@0, D, S]