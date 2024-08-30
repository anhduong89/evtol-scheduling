% revenue if agents fly in the arc (V, V') at time T can serve all passengers of request of arc(V, V') at time T
revenue(N, WR) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR.

% revenue if agents fly in the arc (V, V') at time T can't serve all passengers of request of arc(V, V') at time T
revenue(N, W) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR.

cost(C1) :- C1 = #sum {WR, N: revenue(N, WR)}.

:- cost(C), C < bound.

#show revenue/2.
#show cost/1.