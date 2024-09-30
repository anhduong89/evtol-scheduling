% start time at vertiport X+1 - start time vertiport X = flight time (X, X+1) + charge time at vertiport X+1 
&diff {start(D,E',X+1) - start(D,E,X)} >= T :-  flight_time(E,FT), charge_time(E',CT), T=FT+CT, 
                                                as(D,E,_,X), as(D,E',_,X+1).