% ------ optimization_1: penalize unserved customers.

remain_cust(E,N-P) :- request(_,E,N), P=#sum{W,D,X: as(D,E,W,X)}, N!=P.
:~ remain_cust(E,P). [P@2, E]