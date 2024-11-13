
% revenue if agents fly in the arc (V, V') at time T can serve all passengers of request of arc(V, V') at time T
revenue(N, WR) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR.
% revenue if agents fly in the arc (V, V') at time T can't serve all passengers of request of arc(V, V') at time T
revenue(N, W) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR.

% find edge_w that maximize served passengers
% #maximize{R,N:revenue(N, R)}.
:~ revenue(N, R). [-R@1, N]
% #minimize{C@0, D, X:emission_cost(C, D, X)}.
% #minimize{C@0, D, X:charging_cost(C, D, X)}.



% passengers_served(Q, E) :- Q = #sum{W,D,X:as_w(D, E, X, W)}, request(_, E, _).


#show edge_w/3.
#show node_w/3.
#show distance_t/2.
