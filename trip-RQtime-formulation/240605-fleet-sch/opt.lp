

% revenue if agents fly in the arc (V, V') at time T can serve all passengers of request of arc(V, V') at time T
revenue(N, R/2) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR, distance((V, V'), D), alpha * D * WR = R.
% revenue if agents fly in the arc (V, V') at time T can't serve all passengers of request of arc(V, V') at time T
revenue(N, R/2) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR, distance((V, V'), D), alpha * D * W = R.
% find edge_w that maximize served passengers
#maximize{R,N:revenue(N, R)}.
% #minimize{C@0, D, X:emission_cost(C, D, X)}.
% #minimize{C@0, D, X:charging_cost(C, D, X)}.



% passengers_served(Q, E) :- Q = #sum{W,D,X:as_w(D, E, X, W)}, request(_, E, _).


#show revenue/2.
#show edge_w/3.
#show node_w/3.
#show distance_t/2.