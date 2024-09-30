1{charge(T,D,X):0<=T<=25}1 :- as(D,E,_,X).

battery(B-FT+CT,D,X) :- battery(B,D,X-1), charge(CT,D,X), flight_time(E,FT), as(D,E,_X-1).

:- battery(B,_,_), B < 0.

:- battery(B,_,_), B > 25.


&diff {start(D,E',X+1) - start(D,E,X)} >= T :- charge(CT,D,X), flight_time(E',FT), T=CT+FT,
                                                as(D,E,X), as(D,E',X+1).

em_cost(EMC,D,X) :- charge(CT,D,X), as(D,(V,_),_,X), mer(V,T10,MER), T10-10<=