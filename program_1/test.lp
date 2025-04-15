% a, b, c are number
% a >= 5
% b >= 6
% c >= 3

drone(a;b;c).

duration(a, 5).
duration(b, 6).
duration(c, 3).

charging(X, 2) :- drone(X).

% assign a drone to a gate only
1 {assign(G, D) : gate(G)} 1 :- drone(D).

% can not assign one Gate twice
:- gate(G), gate(G'), G != G', 2 {assign(G, _)}, {assign(G', _)} 0.

gate(1..2).

% the order of gate's assignment of drone
1 {before(D, D'); before(D', D)} 1 :- gate(G), assign(G, D), assign(G, D'), D != D'.

% 
:- before(D, D'), before(D',D''), before(D'', D).

&diff{ 0-start(X) } <= 0 :- drone(X).
&diff{ start(X) - arrival(X) } <= -D :-  drone(X), duration(X, D).
&diff{ arrival(X) - departure(X) } <= -D :-  drone(X), charging(X, D).
&diff{ departure(D) - arrival(D')} <= 0 :- gate(G), assign(G, D), assign(G, D'), before(D, D').

#show.
#show assign/2.
#show dl/2.
#show before/2.

% dl(departure(b),9) dl(arrival(b),7)
% dl(departure(c),5) dl(arrival(c),3)
% dl(departure(a),7) dl(arrival(a),5)
% dl(start(a),0) dl(start(b),0) dl(start(c),0)

% assign(1,c) assign(2,b) assign(2,a)

% before(a,b)